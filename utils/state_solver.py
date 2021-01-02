import argparse
import os
import torch


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

def main():
    parser = argparse.ArgumentParser(description="Remove the solver states stored in a trained model")
    parser.add_argument(
        "--model",
        default="zednet.tar",
        help="path to the input model file",
    )

    args = parser.parse_args()
    model = torch.load(args.model)
    # state_dict = torch.load(args.model)

    # checking item in model
    # print(model)
    # for k,v in model['state_dict'].items():
    #     print(k)
    
    # exit()

    # for k,v in model['state_dict'].items():
    # 	print(k)   
    # exit()


    newModel = model
    newModel['state_dict'] = removekey(newModel['state_dict'],['module.fc.weight','module.fc.bias'])
    del newModel["epoch"]
    del newModel["arch"]
    del newModel["best_acc1"]
    del newModel["optimizer"]



    # del model["optimizer"]
    # del model["scheduler"]
    # del model["iteration"]

    filename_wo_ext, ext = os.path.splitext(args.model)
    output_file = filename_wo_ext + "state_dict" + ext
    torch.save(model, output_file)
    print("Done. The model without solver states is saved to {}".format(output_file))

if __name__ == "__main__":
    main()