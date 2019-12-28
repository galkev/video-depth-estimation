import torch
import argparse

# noinspection PyUnresolvedReferences
import pathmagic
from tools import project
from tools.project import proj_dir
from trainer.trainer import Trainer
from trainer.model_loader import ModelLoader


def main():
    device = "cuda:0"

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data_type", default="test")

    args = parser.parse_args()

    model_path = proj_dir("models", "ddff_mdff")

    loader = ModelLoader(model_path, args.model)
    model_params = loader.load("params")

    model = project.create_component("net", **model_params["net"])

    epoch = args.checkpoint

    if args.checkpoint is not None:
        print("Load from checkpoint")
        model.load_state_dict(loader.load("checkpoint", epoch=epoch)["model"])
    else:
        print("Load model")
        loader.load("model", model)

    # data_class = datatype_from_str(args.dataset)

    #data = data_class(root_dir=proj_dir("datasets"), data_type=args.data_type)

    data = project.create_component(
        "data", name=args.dataset, root_dir=proj_dir("datasets"), data_type=args.data_type)

    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    trainer = Trainer(model, device)

    test_loss, eval_pred = trainer.test_model(data_loader)

    loader.save("eval", {"pred": eval_pred}, epoch=epoch, data_name=args.dataset, data_type=args.data_type)

    print("Done. Loss:", test_loss)


if __name__ == "__main__":
    main()

"""
def eval_models(model_class, model_paths, preload_data=True):
    for i, model_path in enumerate(model_paths):
        print("Eval {}/{}".format(i+1, len(model_paths)))
        print("Model:", model_path)

        test_data = MdffData(proj_dir("datasets"), "test", preload_data=preload_data)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

        eval_model(model_class, model_path, test_loader)


def eval_model(model_class, model_path, data_loader, skip_evaled=True):
    device = "cuda:0"

    model = model_class(load_pretrained=False, **model_loader.load_params(model_path)["model_args"])
    model_loader.load(model_path, model, load_model_stats=False)

    model = model.to(device)

    test_loss, eval_data = Trainer(device=device).test_model(model, data_loader)

    model_loader.save(model_path,
                      eval_data={"loss": test_loss, "eval_data": eval_data})

    print("Test loss: {}".format(test_loss))
    print("Done eval")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", action='store_true')  # eval all models
    parser.add_argument("-r", action='store_true')
    parser.add_argument("--mtype", default="ddff")
    parser.add_argument("--dtype", default="mdff")
    parser.add_argument("models", nargs="*")

    args = parser.parse_args()

    path = proj_dir("models", "ddff_mdff")

    if args.a:
        model_paths = [os.path.dirname(m) for m in
                       glob.iglob(os.path.join(path, "**", "*.pth"), recursive=True)]
    else:
        if len(args.models) == 0:
            print("Error: no models specified")
            exit(1)

        if args.r:
            model_paths = []
            for model in args.models:
                model_paths += [os.path.dirname(m) for m in
                                glob.iglob(os.path.join(path, model, "**", "model.pth"), recursive=True)]
        else:
            model_paths = os.path.join(args.models)

    model_class = modeltype_from_str(args.mtype)

    eval_models(model_class, model_paths)


if __name__ == "__main__":
    main()
"""
