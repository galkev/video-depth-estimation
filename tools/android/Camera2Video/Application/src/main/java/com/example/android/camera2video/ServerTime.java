package com.example.android.camera2video;

import android.util.Log;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

public class ServerTime {
    private static final String TAG = "MYLOG";

    public static final String TIME_FMT_HHMMSSMS = "HH:mm:ss:SSS";

    long server_timestamp_ms = -1;
    long device_timestamp_ms = -1;

    public boolean inSync()
    {
        return server_timestamp_ms != -1;
    }

    public long now()
    {
        return System.currentTimeMillis() - getTimeDiff();
    }

    public long getServerTime()
    {
        return server_timestamp_ms;
    }

    public long getTimeDiff()
    {
        return device_timestamp_ms - server_timestamp_ms;
    }

    public void sync(long server_time, long dev_time) {
        server_timestamp_ms = server_time;
        device_timestamp_ms = dev_time;
    }

    public void logInfo() {

        Log.d(TAG, "Server Time: " + formatTime(server_timestamp_ms, TIME_FMT_HHMMSSMS));
        Log.d(TAG, "Server Time Diff: " + (device_timestamp_ms - server_timestamp_ms) + "ms");
    }

    public static String formatTime(long time_ms, String fmt) {
        Date date = new Date(time_ms);
        DateFormat formatter = new SimpleDateFormat(fmt);
        //formatter.setTimeZone(TimeZone.getTimeZone("UTC"));
        return formatter.format(date);
    }
}
