package com.example.android.camera2video;

import android.app.Activity;
import android.content.Context;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.util.Log;
import android.util.Range;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;


public class FocusControl
{
    class RecordData {
        public class Frame {
            public float focusDist;
            public String time;
            public long timestamp_ms;
            public float requestedFocusDist;

            public Frame(float focDist, long frame_timestamp, float requestedFocDist) {
                focusDist = focDist;
                timestamp_ms = frame_timestamp;
                requestedFocusDist = requestedFocDist;

                time = ServerTime.formatTime(timestamp_ms, ServerTime.TIME_FMT_HHMMSSMS);
            }
        }

        private Range<Float> focusRange = new Range<>(-1.0f, -1.0f);
        private ArrayList<Frame> frames = new ArrayList<>();

        public RecordData() {

        }

        public void addFrame(Frame f) {
            frames.add(f);
        }

        public void init(Range<Float> focRange) {
            reset();
            focusRange = focRange;
        }

        public void debugCheck()
        {
            if (focusRange.getLower() == Float.POSITIVE_INFINITY)
                Log.d(TAG, "focusRange.lower is INF");
            if (focusRange.getUpper() == Float.POSITIVE_INFINITY)
                Log.d(TAG, "focusRange.upper is INF");
            {
                for (int i = 0; i < frames.size(); i++){
                    if (frames.get(i).focusDist == Float.POSITIVE_INFINITY)
                        Log.d(TAG, "Frame " + i + "focusDist is INF");
                    if (frames.get(i).timestamp_ms == Float.POSITIVE_INFINITY)
                        Log.d(TAG, "Frame " + i + "timeDeltaSec is INF");
                    if (frames.get(i).requestedFocusDist == Float.POSITIVE_INFINITY)
                        Log.d(TAG, "Frame " + i + "requestedFocusDist is INF");
                }
            }
        }

        public void reset() {
            frames.clear();
        }

        public int getFrameCount() {
            return frames.size();
        }
    }

    public enum FocusControlFunctions {
        FCF_COSINE,
        FCF_LINEAR
    }

    private static final String TAG = "MYLOG";

    private float focusStartMeter;
    private float focusEndMeter;

    private float timeScale = 1.0f;

    private long startTime = -1;

    private float curRequestedFocusDistMeter = -1.0f;

    private String videoFilePath = "";

    private Activity activity = null;

    private RecordData recordData = new RecordData();

    private FocusControlFunctions focusFunc = FocusControlFunctions.FCF_COSINE;

    public float getFocusStartMeter() {
        return focusStartMeter;
    }

    public float getFocusEndMeter() {
        return focusEndMeter;
    }

    public static float meterToDiopter(float meter)
    {
        return 1.0f / meter;
    }

    public static float diopterToMeter(float diopter)
    {
        return 1.0f / diopter;
    }

    public int getFrameCount() {
        return recordData.getFrameCount();
    }

    public FocusControl(Activity act)
    {
        //setRangeMeter(0.5f, 1.0f);
        activity = act;
        setRangeMeter(0.1f, 1.0f);
    }

    public String getParamsFilePath() {
        return videoFilePath.substring(0, videoFilePath.lastIndexOf('.')) + ".json";
    }

    public void setVideoFilePath(String path) {
        videoFilePath = path;
    }

    public void setTimeScale(float scale) {
        timeScale = scale;
    }

    public void setTimeFunc(FocusControlFunctions func) {
        focusFunc = func;
    }

    public boolean isRunning()
    {
        return startTime != -1;
    }

    public void start()
    {
        startTime = System.currentTimeMillis();

        recordData.init(new Range<>(focusStartMeter, focusEndMeter));
    }

    public void stop()
    {
        //logSummary();
        saveParamsFile();
        startTime = -1;
    }

    public void saveParamsFile()
    {
        recordData.debugCheck();

        GsonBuilder gsonBuilder = new GsonBuilder();
        Gson gson = gsonBuilder.serializeSpecialFloatingPointValues().create();

        String paramsFilePath = getParamsFilePath();
        String paramsFileContents = gson.toJson(recordData);

        File file = new File(paramsFilePath);

        try {
            FileOutputStream stream = new FileOutputStream(file);
            stream.write(paramsFileContents.getBytes());
            stream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public float getCurrentTimeDeltaSec()
    {
        if (isRunning())
            return (System.currentTimeMillis() - startTime) / 1000.0f;
        else
            return 0.0f;
    }

    public static float timeFuncCosine(float t)
    {
        return -(float)Math.cos(t * 2*Math.PI) * 0.5f + 0.5f;
    }

    public static float timeFuncLinear(float t) {
        float tNormalized = 2.0f * (t - (float)Math.floor(t)) - 1.0f;

        return -Math.abs(tNormalized) + 1.0f;
    }

    public float timeFunc(float t) {
        switch (focusFunc) {
            case FCF_COSINE:
                return timeFuncCosine(t);
            case FCF_LINEAR:
                return timeFuncLinear(t);
        }

        return 0.0f;
    }

    public float getFocusDistMeterFromTime(float t)
    {
        float a = timeFunc(timeScale * t);
        return (1.0f - a) * focusStartMeter + a * focusEndMeter;
    }

    // S7 -> 10 diopters / 0.1 meters
    public float getMinFocusDistDiopter(CameraCharacteristics camChars)
    {
        return camChars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
    }

    public void setRangeMeter(float start, float end)
    {
        focusStartMeter = start;
        focusEndMeter = end;
    }

    public void setRangeDiopter(float start, float end)
    {
        setRangeMeter(diopterToMeter(start), diopterToMeter(end));
    }

    public float update(CaptureRequest.Builder builder)
    {
        float curTime = getCurrentTimeDeltaSec();

        curRequestedFocusDistMeter = getFocusDistMeterFromTime(curTime);

        //Log.d(TAG, "Time: " + curTime);
        //Log.d(TAG, "Set Focus: " + curFocusDistMeter);

        builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF);
        builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, meterToDiopter(curRequestedFocusDistMeter));

        return timeScale * curTime;
    }

    public void addFrameInfo(long timestamp, float focusDist)
    {
        recordData.addFrame(recordData.new Frame(
                focusDist,
                timestamp,
                curRequestedFocusDistMeter)
        );
    }

    /*public void logSummary()
    {
        Log.d(TAG, "Record time: " + getCurrentTimeDeltaSec());
        Log.d(TAG, "Frames: " + actualFocalDists.size());

        float maxDiff = 0;

        for (int i = 0; i < actualFocalDists.size(); i++)
        {
            float distDiff = actualFocalDists.get(i) - requestedFocusDists.get(i);
            Log.d(TAG, "Act Focus Dist: " + actualFocalDists.get(i) +
                    "; Req Focus Dist:" + requestedFocusDists.get(i) +
                    "; Diff: " + distDiff);

            if (Math.abs(distDiff) > Math.abs(maxDiff))
                maxDiff = distDiff;
        }

        Log.d(TAG, "Max diff: " + maxDiff);
    }*/
}
