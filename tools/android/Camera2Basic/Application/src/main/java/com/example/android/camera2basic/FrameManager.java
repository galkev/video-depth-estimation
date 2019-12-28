package com.example.android.camera2basic;

import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.ArrayList;

public class FrameManager {
    class FrameInfo {
        float focDist;
        long timeMs;

        public FrameInfo(long timeMs, float focDist) {
            this.focDist = focDist;
            this.timeMs = timeMs;
        }
    }

    class FrameParamsJson {
        class Frame {
            int idx;
            float focDist;

            public Frame(int idx, float focDist) {
                this.idx = idx;
                this.focDist = focDist;
            }
        }

        float focusRangeStart;
        float focusRangeEnd;

        Frame[] frames;

        public FrameParamsJson(float focusRangeStart, float focusRangeEnd, FrameInfo[] frameInfos) {
            this.focusRangeStart = focusRangeStart;
            this.focusRangeEnd = focusRangeEnd;

            frames = new Frame[frameInfos.length];
            for (int i = 0; i < frameInfos.length; i++) {
                frames[i] = new Frame(i, frameInfos[i].focDist);
            }
        }
    }

    private static final String TAG = "Camera2BasicFragment";

    File mSeqDir;

    ArrayList<FrameInfo> mFrameInfos = new ArrayList<>();
    ArrayList<Float> mRequestedFocusDists = new ArrayList<>();

    public FrameManager(File seqDir) {
        mSeqDir = seqDir;
    }

    public static float meterToDiopter(float meter) {
        return 1.0f / meter;
    }

    public static float diopterToMeter(float diopter) {
        return 1.0f / diopter;
    }

    public void addFrame(float reqFocusDist, TotalCaptureResult result) {
        mRequestedFocusDists.add(reqFocusDist);
        //long frameTimeMs = System.currentTimeMillis();
        long frameTimeMs = result.get(CaptureResult.SENSOR_TIMESTAMP) / 1000000;

        //if (mCurFrameIndex == 0)
        //    firstFrameTime = frameTimeMilli;

        float foc_dist = diopterToMeter(result.get(CaptureResult.LENS_FOCUS_DISTANCE));

        mFrameInfos.add(new FrameInfo(frameTimeMs, foc_dist));

        // Log.d(TAG, "Burst " + mCurFrameIndex + " - FocDist: " + foc_dist);
    }

    public void saveJson(File file, FrameParamsJson data) {
        GsonBuilder gsonBuilder = new GsonBuilder();
        Gson gson = gsonBuilder.serializeSpecialFloatingPointValues().setPrettyPrinting().create();

        String paramsFileContents = gson.toJson(data);

        try {
            FileOutputStream stream = new FileOutputStream(file);
            stream.write(paramsFileContents.getBytes());
            stream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void save() {
        String logMsg = " \nReq Focus Dist | Focus Dist | Delta Time | Total Time\n";

        for (int i = 0; i < mFrameInfos.size(); i++) {
            FrameInfo frameInfo = mFrameInfos.get(i);

            long delta = (i > 0) ? frameInfo.timeMs - mFrameInfos.get(i - 1).timeMs : 0;
            long time = frameInfo.timeMs - mFrameInfos.get(0).timeMs;

            logMsg += String.format("%f, %f, %dms, %dms\n",
                    mRequestedFocusDists.get(i), frameInfo.focDist, delta, time);
        }

        Log.d(TAG, logMsg);

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(mSeqDir, "capture.txt")));
            writer.write(logMsg);
            writer.close();

            FrameParamsJson params = new FrameParamsJson(
                    mRequestedFocusDists.get(0),
                    mRequestedFocusDists.get(mRequestedFocusDists.size() - 1),
                    mFrameInfos.toArray(new FrameInfo[mFrameInfos.size()])
            );

            saveJson(new File(mSeqDir, "params.json"), params);
        }
        catch (IOException e) {
            Log.d(TAG, "Error writing log");
        }
    }
}
