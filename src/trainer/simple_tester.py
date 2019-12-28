import torch


class SimpleTester(object):
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)

    def load_checkpoint(self, checkpoint, mode="train"):
        self.model.load_state_dict(checkpoint["model"])
        print("SimpleTester loaded checkpoint")

    def test(self, test_loader, out_idx=-1):
        self.model.eval()

        with torch.no_grad():
            outputs = []

            for test_it, data in enumerate(test_loader, 1):
                out = self.model(data.to(self.device))

                if not isinstance(out, torch.Tensor):
                    out = out[out_idx]

                outputs.append(out)
                print(f"Test {test_it}")

            print("Done")

            outputs = torch.cat(outputs)
            return outputs
