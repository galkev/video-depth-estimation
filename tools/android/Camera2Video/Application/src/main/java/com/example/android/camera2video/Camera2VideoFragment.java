/*
 * Copyright 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camera2video;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.app.Fragment;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.v13.app.FragmentCompat;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.util.SizeF;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.instacart.library.truetime.TrueTimeRx;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import io.reactivex.schedulers.Schedulers;

public class Camera2VideoFragment extends Fragment
        implements View.OnClickListener, FragmentCompat.OnRequestPermissionsResultCallback {

    static String ntpServer = "time.google.com";

    private static final int SENSOR_ORIENTATION_DEFAULT_DEGREES = 90;
    private static final int SENSOR_ORIENTATION_INVERSE_DEGREES = 270;
    private static final SparseIntArray DEFAULT_ORIENTATIONS = new SparseIntArray();
    private static final SparseIntArray INVERSE_ORIENTATIONS = new SparseIntArray();

    private static final String TAG = "MYLOG";
    private static final int REQUEST_VIDEO_PERMISSIONS = 1;
    private static final String FRAGMENT_DIALOG = "dialog";

    private static final String[] VIDEO_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
    };

    static {
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_0, 90);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_90, 0);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_180, 270);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    static {
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_0, 270);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_90, 180);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_180, 90);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_270, 0);
    }

    /**
     * An {@link AutoFitTextureView} for camera preview.
     */
    private AutoFitTextureView mTextureView;

    /**
     * Button to record video
     */
    private Button mButtonVideo;

    private TextView mStatus;

    private float mCurFocDist = -1.0f;

    /**
     * A reference to the opened {@link android.hardware.camera2.CameraDevice}.
     */
    private CameraDevice mCameraDevice;

    /**
     * A reference to the current {@link android.hardware.camera2.CameraCaptureSession} for
     * preview.
     */
    private CameraCaptureSession mPreviewSession;

    /**
     * {@link TextureView.SurfaceTextureListener} handles several lifecycle events on a
     * {@link TextureView}.
     */
    private TextureView.SurfaceTextureListener mSurfaceTextureListener
            = new TextureView.SurfaceTextureListener() {

        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture,
                                              int width, int height) {
            openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture,
                                                int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
        }

    };

    /**
     * The {@link android.util.Size} of camera preview.
     */
    private Size mPreviewSize;

    /**
     * The {@link android.util.Size} of video recording.
     */
    private Size mVideoSize;

    /**
     * MediaRecorder
     */
    private MediaRecorder mMediaRecorder;

    /**
     * Whether the app is recording video now
     */
    private boolean mIsRecordingVideo;

    private int mCurFpsRangeIdx = 0;
    private int mCurResSizeIdx = 0;

    private int mRuntimeModeId = 0;

    /**
     * An additional thread for running tasks that shouldn't block the UI.
     */
    private HandlerThread mBackgroundThread;

    /**
     * A {@link Handler} for running tasks in the background.
     */
    private Handler mBackgroundHandler;

    /**
     * A {@link Semaphore} to prevent the app from exiting before closing the camera.
     */
    private Semaphore mCameraOpenCloseLock = new Semaphore(1);

    private FocusControl mFocusControl;

    private final Handler guiHandler = new Handler();
    private final Runnable guiUpdateRunnable = new Runnable() {
        public void run() {
            updateStatus();
        }
    };

    private CameraCharacteristics mCamChars = null;
    long frameTimestamp = 0;

    private CameraCaptureSession.CaptureCallback mCaptureCallback
            = new CameraCaptureSession.CaptureCallback() {

        @Override
        public void onCaptureCompleted(CameraCaptureSession session,
                                       CaptureRequest request,
                                       TotalCaptureResult result)
        {
            /*if (!ttInit) {

                Log.d(TAG, "True time initialized");

                ttInit = true;
            }

            ntpTimestamp = TrueTimeRx.now();*/

            frameTimestamp = serverTime.now();

            if (mCameraDevice != null) {
                mCurFocDist = FocusControl.diopterToMeter(result.get(CaptureResult.LENS_FOCUS_DISTANCE));

                if (mIsRecordingVideo) {
                    mFocusControl.addFrameInfo(frameTimestamp, mCurFocDist);
                }

                try {
                    setRepeatingRequest();
                }
                catch (Exception e) {}
            }
        }
    };

    /**
     * {@link CameraDevice.StateCallback} is called when {@link CameraDevice} changes its status.
     */
    private CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            startPreview();
            mCameraOpenCloseLock.release();
            if (null != mTextureView) {
                configureTransform(mTextureView.getWidth(), mTextureView.getHeight());
            }
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
            Activity activity = getActivity();
            if (null != activity) {
                activity.finish();
            }
        }

    };
    private Integer mSensorOrientation;
    public String mNextVideoAbsolutePath;
    private CaptureRequest.Builder mPreviewBuilder;

    public static Camera2VideoFragment newInstance() {
        return new Camera2VideoFragment();
    }

    /**
     * In this sample, we choose a video size with 3x4 aspect ratio. Also, we don't use sizes
     * larger than 1080p, since MediaRecorder cannot handle such a high-resolution video.
     *
     * @param choices The list of available sizes
     * @return The video size
     */
    private Size chooseVideoSize(Size[] choices) {
        Log.d(TAG, "chooseVideoSize " + choices[mCurResSizeIdx]);
        return choices[mCurResSizeIdx];
        //return new Size(640, 480);
        //return new Size(1280, 720);

        /*for (Size size : choices) {
            if (size.getWidth() == size.getHeight() * 4 / 3 && size.getWidth() <= 1080) {
                return size;
            }
        }
        Log.e(TAG, "Couldn't find any suitable video size");
        return choices[choices.length - 1];*/
    }

    /**
     * Given {@code choices} of {@code Size}s supported by a camera, chooses the smallest one whose
     * width and height are at least as large as the respective requested values, and whose aspect
     * ratio matches with the specified value.
     *
     * @param choices     The list of sizes that the camera supports for the intended output class
     * @param width       The minimum desired width
     * @param height      The minimum desired height
     * @param aspectRatio The aspect ratio
     * @return The optimal {@code Size}, or an arbitrary one if none were big enough
     */
    private static Size chooseOptimalSize(Size[] choices, int width, int height, Size aspectRatio) {
        // Collect the supported resolutions that are at least as big as the preview Surface
        List<Size> bigEnough = new ArrayList<>();
        int w = aspectRatio.getWidth();
        int h = aspectRatio.getHeight();
        for (Size option : choices) {
            if (option.getHeight() == option.getWidth() * h / w &&
                    option.getWidth() >= width && option.getHeight() >= height) {
                bigEnough.add(option);
            }
        }

        // Pick the smallest of those, assuming we found any
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }

    ServerThread serverThread;
    ServerTime serverTime;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        TrueTimeRx.build()
                .initializeRx(ntpServer)
                .subscribeOn(Schedulers.io())
                .subscribe(date -> {
                    Log.v(TAG, "TrueTime was initialized and we have a time: " + date);
                }, throwable -> {
                    throwable.printStackTrace();
                });

        serverThread = new ServerThread(this);
        serverThread.start();

        serverTime = new ServerTime();

        return inflater.inflate(R.layout.fragment_camera2_video, container, false);
    }

    public ServerTime getServerTimer() {
        return serverTime;
    }

    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        mTextureView = (AutoFitTextureView) view.findViewById(R.id.texture);
        mButtonVideo = (Button) view.findViewById(R.id.video);
        mButtonVideo.setOnClickListener(this);
        view.findViewById(R.id.settings).setOnClickListener(this);

        mTextureView.setVisibility(View.VISIBLE);

        mStatus = (TextView)view.findViewById(R.id.status);

        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {updateGui();}
        }, 0, 50);

        Log.d("ServerThread", "onViewCreated");
    }

    public void updateGui() {
        guiHandler.post(guiUpdateRunnable);
    }

    boolean autofocus = false;

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();
        if (mTextureView.isAvailable()) {
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }

        SharedPreferences sharedPref =
                PreferenceManager.getDefaultSharedPreferences(getActivity());

        float focalDistStart = Float.valueOf(sharedPref.getString("foc_dist_start", "0.1"));
        float focalDistEnd = Float.valueOf(sharedPref.getString("foc_dist_end", "1.0"));
        int focusFuncId = Integer.valueOf(sharedPref.getString("foc_time_func", "0"));
        mRuntimeModeId = Integer.valueOf(sharedPref.getString("runtime", "0"));
        float timeScale = Float.valueOf(sharedPref.getString("time_scale", "0.3"));
        autofocus = sharedPref.getBoolean("autofocus", false);

        mCurFpsRangeIdx = Integer.valueOf(sharedPref.getString("target_fps", "0"));
        mCurResSizeIdx = Integer.valueOf(sharedPref.getString("target_res", "0"));

        mFocusControl = new FocusControl(getActivity());
        mFocusControl.setRangeMeter(focalDistStart, focalDistEnd);
        mFocusControl.setTimeFunc(FocusControl.FocusControlFunctions.values()[focusFuncId]);
        mFocusControl.setTimeScale(timeScale);

        Log.d(TAG, "Runtime: " + mRuntimeModeId);
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void printCamInfo() {
        SizeF size = mCamChars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
        float[] focalLengths = mCamChars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        float[] apertures = mCamChars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES);
        Log.d(TAG, "Sensor size: " + size.getWidth() + "x" + size.getHeight());
        Log.d(TAG, "Focal length: " + java.util.Arrays.toString(focalLengths));
        Log.d(TAG, "Apertures: " + java.util.Arrays.toString(apertures));
    }

    private Range<Integer>[] getFpsRanges() {
        if (mCamChars != null)
            return mCamChars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        else
            return null;
    }

    private Size[] getResSizes() {

        if (mCamChars != null) {
            StreamConfigurationMap map = mCamChars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            return map.getOutputSizes(MediaRecorder.class);
        }
        else
            return null;
    }

    private Size getResSize() {
        Size[] res = getResSizes();

        if (res != null)
            return res[mCurResSizeIdx];
        else
            return null;
    }

    private Range getFpsRange() {
        Range<Integer>[] fpsRanges = getFpsRanges();

        if (fpsRanges != null) {
            return fpsRanges[mCurFpsRangeIdx];
        }
        else
            return null;
    }

    private void captureStillPicture() {
        try {
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            mPreviewBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            List<Surface> surfaces = new ArrayList<>();

            // Set up Surface for the camera preview
            Surface previewSurface = new Surface(texture);
            surfaces.add(previewSurface);
            mPreviewBuilder.addTarget(previewSurface);

            // Set up Surface for the MediaRecorder
            Surface recorderSurface = mMediaRecorder.getSurface();
            surfaces.add(recorderSurface);
            mPreviewBuilder.addTarget(recorderSurface);

            List<CaptureRequest> captureList = new ArrayList<CaptureRequest>();

            for (int i=0;i<10;i++) {
                captureList.add(mPreviewBuilder.build());
            }

            mPreviewSession.stopRepeating();
            mPreviewSession.captureBurst(captureList, cameraCaptureCallback, null);
            mPreviewBuilder.removeTarget(recorderSurface);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    CameraCaptureSession.CaptureCallback cameraCaptureCallback = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request,
                                       TotalCaptureResult result) {
            Log.d("MYLOG","saved");
            /*
            mPictureCounter++;
            if (mPictureCounter >= 10)
                unlockFocus();
                */
        }
    };

    public void toggleRecord() {
        if (mIsRecordingVideo) {
            stopRecordingVideo();
        } else {
            startRecordingVideo();
        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.video: {
                toggleRecord();
                break;
            }
            case R.id.settings: {
                if (!mIsRecordingVideo) {
                    // S7 -> [[15, 15], [24, 24], [10, 30], [15, 30], [30, 30]]
                    Range<Integer>[] fpsRanges = getFpsRanges();
                    String[] fpsRangesStr = new String[fpsRanges.length];

                    for (int i = 0; i < fpsRanges.length; i++)
                        fpsRangesStr[i] = fpsRanges[i].toString();

                    Size[] resSizes = getResSizes();
                    String[] resSizesStr = new String[resSizes.length];

                    for (int i = 0; i < resSizes.length; i++)
                        resSizesStr[i] = resSizes[i].toString();

                    Intent i = new Intent(getActivity(), SettingsActivity.class);
                    i.putExtra("fps_ranges", fpsRangesStr);
                    i.putExtra("res_sizes", resSizesStr);
                    startActivity(i);
                }
                /*Activity activity = getActivity();
                if (null != activity) {
                    new AlertDialog.Builder(activity)
                            .setMessage(R.string.intro_message)
                            .setPositiveButton(android.R.string.ok, null)
                            .show();
                }*/
                break;
            }
        }
    }

    /**
     * Starts a background thread and its {@link Handler}.
     */
    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    /**
     * Stops the background thread and its {@link Handler}.
     */
    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Gets whether you should show UI with rationale for requesting permissions.
     *
     * @param permissions The permissions your app wants to request.
     * @return Whether you can show permission rationale UI.
     */
    private boolean shouldShowRequestPermissionRationale(String[] permissions) {
        for (String permission : permissions) {
            if (FragmentCompat.shouldShowRequestPermissionRationale(this, permission)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Requests permissions needed for recording video.
     */
    private void requestVideoPermissions() {
        if (shouldShowRequestPermissionRationale(VIDEO_PERMISSIONS)) {
            new ConfirmationDialog().show(getChildFragmentManager(), FRAGMENT_DIALOG);
        } else {
            FragmentCompat.requestPermissions(this, VIDEO_PERMISSIONS, REQUEST_VIDEO_PERMISSIONS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        Log.d(TAG, "onRequestPermissionsResult");
        if (requestCode == REQUEST_VIDEO_PERMISSIONS) {
            if (grantResults.length == VIDEO_PERMISSIONS.length) {
                for (int result : grantResults) {
                    if (result != PackageManager.PERMISSION_GRANTED) {
                        ErrorDialog.newInstance(getString(R.string.permission_request))
                                .show(getChildFragmentManager(), FRAGMENT_DIALOG);
                        break;
                    }
                }
            } else {
                ErrorDialog.newInstance(getString(R.string.permission_request))
                        .show(getChildFragmentManager(), FRAGMENT_DIALOG);
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.d("ServerThread", "onCreate");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        serverThread.setIsLoop(false);
        serverThread.cancel();
        serverThread.interrupt();

        Log.d("ServerThread", "onDestroy");
    }

    private void updateStatus() {
        if (mFocusControl != null) {
            Range fpsRange = getFpsRange();
            String fpsRangeStr = fpsRange != null ? fpsRange.toString() : "null";

            Size resSize = getResSize();
            String resSizeStr = resSize != null ? resSize.toString() : "null1";

            @SuppressLint("DefaultLocale") String stat = String.format(
                    "FocDist: %.3fm %s, FPS: %s, Res: %s\nRecord: %.3fs, #Frames: %d",
                    mCurFocDist,
                    !autofocus ? String.format("[%.3fm, %.3fm]",
                            mFocusControl.getFocusStartMeter(),
                            mFocusControl.getFocusEndMeter()
                    ) : "Autofocus",
                    fpsRangeStr,
                    resSizeStr,
                    mFocusControl.getCurrentTimeDeltaSec(),
                    mFocusControl.getFrameCount());

            if (serverTime.inSync())
                stat += "\nLast server time: " +
                        ServerTime.formatTime(serverTime.getServerTime(), ServerTime.TIME_FMT_HHMMSSMS)
                        + "; Diff: " + serverTime.getTimeDiff() + "ms";
            else
                stat += "\nNo server time. Don't record!!!!!!";

            mStatus.setText(stat);
        }
    }

    private boolean hasPermissionsGranted(String[] permissions) {
        for (String permission : permissions) {
            if (ActivityCompat.checkSelfPermission(getActivity(), permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    /**
     * Tries to open a {@link CameraDevice}. The result is listened by `mStateCallback`.
     */
    @SuppressWarnings("MissingPermission")
    private void openCamera(int width, int height) {
        if (!hasPermissionsGranted(VIDEO_PERMISSIONS)) {
            requestVideoPermissions();
            return;
        }
        final Activity activity = getActivity();
        if (null == activity || activity.isFinishing()) {
            return;
        }
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            Log.d(TAG, "tryAcquire");
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            String cameraId = manager.getCameraIdList()[0];

            // Choose the sizes for camera preview and video recording
            mCamChars = manager.getCameraCharacteristics(cameraId);

            printCamInfo();

            StreamConfigurationMap map = mCamChars
                    .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            mSensorOrientation = mCamChars.get(CameraCharacteristics.SENSOR_ORIENTATION);
            if (map == null) {
                throw new RuntimeException("Cannot get available preview/video sizes");
            }
            mVideoSize = chooseVideoSize(map.getOutputSizes(MediaRecorder.class));
            mPreviewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class),
                    width, height, mVideoSize);

            int orientation = getResources().getConfiguration().orientation;
            if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
                mTextureView.setAspectRatio(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            } else {
                mTextureView.setAspectRatio(mPreviewSize.getHeight(), mPreviewSize.getWidth());
            }
            configureTransform(width, height);
            mMediaRecorder = new MediaRecorder();
            manager.openCamera(cameraId, mStateCallback, null);
        } catch (CameraAccessException e) {
            Toast.makeText(activity, "Cannot access the camera.", Toast.LENGTH_SHORT).show();
            activity.finish();
        } catch (NullPointerException e) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            ErrorDialog.newInstance(getString(R.string.camera_error))
                    .show(getChildFragmentManager(), FRAGMENT_DIALOG);
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.");
        }
    }

    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            closePreviewSession();
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
            if (null != mMediaRecorder) {
                mMediaRecorder.release();
                mMediaRecorder = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.");
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    /**
     * Start the camera preview.
     */
    private void startPreview() {
        if (null == mCameraDevice || !mTextureView.isAvailable() || null == mPreviewSize) {
            return;
        }
        try {
            closePreviewSession();
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            mPreviewBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);

            Surface previewSurface = new Surface(texture);
            mPreviewBuilder.addTarget(previewSurface);

            mCameraDevice.createCaptureSession(Collections.singletonList(previewSurface),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            mPreviewSession = session;
                            updatePreview();
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Activity activity = getActivity();
                            if (null != activity) {
                                Toast.makeText(activity, "Failed", Toast.LENGTH_SHORT).show();
                            }
                        }
                    }, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    /**
     * Update the camera preview. {@link #startPreview()} needs to be called in advance.
     */
    private void updatePreview() {
        if (null == mCameraDevice) {
            return;
        }
        HandlerThread thread = new HandlerThread("CameraPreview");

        thread.start();

        setRepeatingRequest();
    }

    private void setRepeatingRequest()
    {
        if (mPreviewSession != null) {
            try {
                if (setUpCaptureRequestBuilder(mPreviewBuilder))
                    mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), mCaptureCallback, mBackgroundHandler);
                else
                    stopRecordingVideo();
            } catch (CameraAccessException e) {
                e.printStackTrace();
            } catch (Exception e) {
                Log.e(TAG, "Failed to start camera preview.", e);
            }
        }
    }

    private boolean setUpCaptureRequestBuilder(CaptureRequest.Builder builder) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);

        Range fpsRange = getFpsRange();

        if (fpsRange != null)
            builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);

        if (!autofocus) {
            float numRamps = mFocusControl.update(builder);

            if (mRuntimeModeId == 1 && numRamps > 0.5)
                return false;
        }
        //else
            //builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO);

        return true;
    }

    /**
     * Configures the necessary {@link android.graphics.Matrix} transformation to `mTextureView`.
     * This method should not to be called until the camera preview size is determined in
     * openCamera, or until the size of `mTextureView` is fixed.
     *
     * @param viewWidth  The width of `mTextureView`
     * @param viewHeight The height of `mTextureView`
     */
    private void configureTransform(int viewWidth, int viewHeight) {
        Activity activity = getActivity();
        if (null == mTextureView || null == mPreviewSize || null == activity) {
            return;
        }
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale = Math.max(
                    (float) viewHeight / mPreviewSize.getHeight(),
                    (float) viewWidth / mPreviewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        }
        mTextureView.setTransform(matrix);
    }

    private void setUpMediaRecorder() throws IOException {
        final Activity activity = getActivity();
        if (null == activity) {
            return;
        }
        mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mMediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        if (mNextVideoAbsolutePath == null || mNextVideoAbsolutePath.isEmpty()) {
            mNextVideoAbsolutePath = getVideoFilePath(getActivity());
        }
        mMediaRecorder.setOutputFile(mNextVideoAbsolutePath);
        mMediaRecorder.setVideoEncodingBitRate(10000000);
        mMediaRecorder.setVideoFrameRate(30);
        Log.d(TAG, "setVideoSize " + mVideoSize.toString());
        mMediaRecorder.setVideoSize(mVideoSize.getWidth(), mVideoSize.getHeight());
        mMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        switch (mSensorOrientation) {
            case SENSOR_ORIENTATION_DEFAULT_DEGREES:
                mMediaRecorder.setOrientationHint(DEFAULT_ORIENTATIONS.get(rotation));
                break;
            case SENSOR_ORIENTATION_INVERSE_DEGREES:
                mMediaRecorder.setOrientationHint(INVERSE_ORIENTATIONS.get(rotation));
                break;
        }
        mMediaRecorder.prepare();
    }

    private String getVideoFilePath(Context context) {
        final File dir = context.getExternalFilesDir(null);
        return (dir == null ? "" : (dir.getAbsolutePath() + "/"))
                + System.currentTimeMillis() + ".mp4";
    }

    public void startRecordingVideo() {
        if (null == mCameraDevice || !mTextureView.isAvailable() || null == mPreviewSize) {
            return;
        }
        try {
            closePreviewSession();
            setUpMediaRecorder();

            mFocusControl.setVideoFilePath(mNextVideoAbsolutePath);

            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            mPreviewBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            List<Surface> surfaces = new ArrayList<>();

            // Set up Surface for the camera preview
            Surface previewSurface = new Surface(texture);
            surfaces.add(previewSurface);
            mPreviewBuilder.addTarget(previewSurface);

            // Set up Surface for the MediaRecorder
            Surface recorderSurface = mMediaRecorder.getSurface();
            surfaces.add(recorderSurface);
            mPreviewBuilder.addTarget(recorderSurface);

            // Start a capture session
            // Once the session starts, we can update the UI and start recording
            mCameraDevice.createCaptureSession(surfaces, new CameraCaptureSession.StateCallback() {

                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    mPreviewSession = cameraCaptureSession;
                    updatePreview();
                    getActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // UI
                            mIsRecordingVideo = true;
                            mButtonVideo.setText("Stop");

                            // Start recording
                            mMediaRecorder.start();

                            mFocusControl.start();
                        }
                    });
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Activity activity = getActivity();
                    if (null != activity) {
                        Toast.makeText(activity, "Failed", Toast.LENGTH_SHORT).show();
                    }
                }
            }, mBackgroundHandler);
        } catch (CameraAccessException | IOException e) {
            e.printStackTrace();
        }

    }

    private void closePreviewSession() {
        if (mPreviewSession != null) {
            mPreviewSession.close();
            mPreviewSession = null;
        }
    }

    public void stopRecordingVideo() {
        // UI
        mIsRecordingVideo = false;
        mButtonVideo.setText("Record");

        // Stop recording
        mMediaRecorder.stop();
        mMediaRecorder.reset();

        mFocusControl.stop();

        Activity activity = getActivity();

        if (null != activity) {
            Toast.makeText(activity, "Video saved: " + mNextVideoAbsolutePath,
                    Toast.LENGTH_SHORT).show();
            Log.d(TAG, "Video saved: " + mNextVideoAbsolutePath);
        }


        mNextVideoAbsolutePath = null;

        startPreview();
    }

    /**
     * Compares two {@code Size}s based on their areas.
     */
    static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }

    }

    public static class ErrorDialog extends DialogFragment {

        private static final String ARG_MESSAGE = "message";

        public static ErrorDialog newInstance(String message) {
            ErrorDialog dialog = new ErrorDialog();
            Bundle args = new Bundle();
            args.putString(ARG_MESSAGE, message);
            dialog.setArguments(args);
            return dialog;
        }

        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Activity activity = getActivity();
            return new AlertDialog.Builder(activity)
                    .setMessage(getArguments().getString(ARG_MESSAGE))
                    .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialogInterface, int i) {
                            activity.finish();
                        }
                    })
                    .create();
        }

    }

    public static class ConfirmationDialog extends DialogFragment {

        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Fragment parent = getParentFragment();
            return new AlertDialog.Builder(getActivity())
                    .setMessage(R.string.permission_request)
                    .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            FragmentCompat.requestPermissions(parent, VIDEO_PERMISSIONS,
                                    REQUEST_VIDEO_PERMISSIONS);
                        }
                    })
                    .setNegativeButton(android.R.string.cancel,
                            new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    parent.getActivity().finish();
                                }
                            })
                    .create();
        }

    }

}
