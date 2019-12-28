"""
class TrainSetupBlenderLSTM(TrainSetupBlender):
    def __init__(self,
                 lr=1e-4,
                 batch_size=6,
                 num_epochs=10000000,
                 focal_stack_size=10,
                 dropout=0.0,
                 checkpoint_freq=3000,  # factor of 450 compared to DDFF
                 lstm_dim=2048,
                 lstm_last_frame_only=False
                 ):
        super().__init__(lr,
                         batch_size,
                         num_epochs,
                         focal_stack_size,
                         dropout,
                         checkpoint_freq)

        self.lstm_dim = lstm_dim
        self.lstm_last_frame_only = lstm_last_frame_only

    def create_model(self):
        return DDFFNetRNN(
            focal_stack_size=self.focal_stack_size,
            dropout=self.dropout,
            lstm_dim=self.lstm_dim,
            lstm_last_frame_only=self.lstm_last_frame_only
        )

    def create_dataset(self, dataset_path, data_type):
        data = VideoDepthFocusData(dataset_path, data_type, "dining_room")
        data.configure(depth_output_indices=-1)
        return data
"""
