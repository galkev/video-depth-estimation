package com.example.android.camera2video;

import android.app.Activity;
import android.os.Bundle;
import android.os.Message;
import android.util.Log;
import android.widget.Toast;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

// https://github.com/gavinliu/Android-Pc-Socket-Connection/blob/master/app/src/main/java/cn/gavinliu/android_pc_socket_connection/MainActivity.java
class ServerThread extends Thread {
    ServerSocket serverSocket = null;

    static String TAG = "MYLOG";
    boolean isLoop = true;
    Camera2VideoFragment parent;

    public ServerThread(Camera2VideoFragment frag) {
        super();
        parent = frag;
    }

    public void setIsLoop(boolean isLoop) {
        this.isLoop = isLoop;
    }

    private void processMessage(String msg, long dev_time) {
        Log.d(TAG, msg);
        String[] msg_parts = msg.split(";");

        Activity activity = parent.getActivity();
        switch (msg_parts[0]) {
            case "time":
                long server_timestamp_ms = Long.parseLong(msg_parts[1]);
                long expected_delay = Long.parseLong(msg_parts[2]);

                server_timestamp_ms += expected_delay;

                parent.getServerTimer().sync(server_timestamp_ms, dev_time);
                parent.getServerTimer().logInfo();
                break;
            case "start_rec":
                activity.runOnUiThread(new Runnable() {
                    public void run() {
                        parent.startRecordingVideo();
                    }
                });
                break;
            case "stop_rec":
                activity.runOnUiThread(new Runnable() {
                    public void run() {
                        parent.stopRecordingVideo();
                    }
                });
                break;
            default:
                Log.d(TAG, "Unrecognized cmd");
        }
    }

    public void cancel()
    {
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void run() {
        Log.d(TAG, "running");

        try {
            serverSocket = new ServerSocket(9000);
            Log.d(TAG, "Thread created");

            while (isLoop) {
                Socket socket = serverSocket.accept();

                Log.d(TAG, "accept");

                DataInputStream inputStream = new DataInputStream(socket.getInputStream());
                DataOutputStream outputStream = new DataOutputStream(socket.getOutputStream());

                ArrayList<String> messages = new ArrayList<>();
                String recv_msg;
                long dev_time = 0;

                while ((recv_msg = inputStream.readLine()) != null)
                {
                    dev_time = System.currentTimeMillis();
                    outputStream.write("ok".getBytes());

                    messages.add(recv_msg);
                }

                socket.close();

                for (String msg : messages)
                    processMessage(msg, dev_time);
            }

        } catch (IOException e) {
            Log.d(TAG, "exception", e);
        } finally {
            Log.d(TAG, "destory");

            if (serverSocket != null) {
                try {
                    serverSocket.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
