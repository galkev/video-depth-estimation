import torch


def map_state_dict(vgg16_features_dict, bias):
    layer_mappings = {'0.weight': 'conv1_1.weight',
            '2.weight': 'conv1_2.weight',
            '5.weight': 'conv2_1.weight',
            '7.weight': 'conv2_2.weight',
            '10.weight': 'conv3_1.weight',
            '12.weight': 'conv3_2.weight',
            '14.weight': 'conv3_3.weight',
            '17.weight': 'conv4_1.weight',
            '19.weight': 'conv4_2.weight',
            '21.weight': 'conv4_3.weight',
            '24.weight': 'conv5_1.weight',
            '26.weight': 'conv5_2.weight',
            '28.weight': 'conv5_3.weight'}
    if bias:
        layer_mappings.update({'0.bias': 'conv1_1.bias',
            '2.bias': 'conv1_2.bias',
            '5.bias': 'conv2_1.bias',
            '7.bias': 'conv2_2.bias',
            '10.bias': 'conv3_1.bias',
            '12.bias': 'conv3_2.bias',
            '14.bias': 'conv3_3.bias',
            '17.bias': 'conv4_1.bias',
            '19.bias': 'conv4_2.bias',
            '21.bias': 'conv4_3.bias',
            '24.bias': 'conv5_1.bias',
            '26.bias': 'conv5_2.bias',
            '28.bias': 'conv5_3.bias'})
    #Update according to generated mapping
    pretrained_dict = {layer_mappings[k]: v for k, v in vgg16_features_dict.items() if k in layer_mappings}
    return pretrained_dict


def map_state_dict_bn(vgg16_features_dict, bias):
    layer_mappings = {'0.weight': 'conv1_1.weight',
            '1.weight': 'conv1_1_bn.weight', '1.bias': 'conv1_1_bn.bias', '1.running_mean': 'conv1_1_bn.running_mean', '1.running_var': 'conv1_1_bn.running_var',
            '3.weight': 'conv1_2.weight',
            '4.weight': 'conv1_2_bn.weight', '4.bias': 'conv1_2_bn.bias', '4.running_mean': 'conv1_2_bn.running_mean', '4.running_var': 'conv1_2_bn.running_var',
            '7.weight': 'conv2_1.weight',
            '8.weight': 'conv2_1_bn.weight', '8.bias': 'conv2_1_bn.bias', '8.running_mean': 'conv2_1_bn.running_mean', '8.running_var': 'conv2_1_bn.running_var',
            '10.weight': 'conv2_2.weight',
            '11.weight': 'conv2_2_bn.weight', '11.bias': 'conv2_2_bn.bias', '11.running_mean': 'conv2_2_bn.running_mean', '11.running_var': 'conv2_2_bn.running_var',
            '14.weight': 'conv3_1.weight',
            '15.weight': 'conv3_1_bn.weight', '15.bias': 'conv3_1_bn.bias', '15.running_mean': 'conv3_1_bn.running_mean', '15.running_var': 'conv3_1_bn.running_var',
            '17.weight': 'conv3_2.weight',
            '18.weight': 'conv3_2_bn.weight', '18.bias': 'conv3_2_bn.bias', '18.running_mean': 'conv3_2_bn.running_mean', '18.running_var': 'conv3_2_bn.running_var',
            '20.weight': 'conv3_3.weight',
            '21.weight': 'conv3_3_bn.weight', '21.bias': 'conv3_3_bn.bias', '21.running_mean': 'conv3_3_bn.running_mean', '21.running_var': 'conv3_3_bn.running_var',
            '24.weight': 'conv4_1.weight',
            '25.weight': 'conv4_1_bn.weight', '25.bias': 'conv4_1_bn.bias', '25.running_mean': 'conv4_1_bn.running_mean', '25.running_var': 'conv4_1_bn.running_var',
            '27.weight': 'conv4_2.weight',
            '28.weight': 'conv4_2_bn.weight', '28.bias': 'conv4_2_bn.bias', '28.running_mean': 'conv4_2_bn.running_mean', '28.running_var': 'conv4_2_bn.running_var',
            '30.weight': 'conv4_3.weight',
            '31.weight': 'conv4_3_bn.weight', '31.bias': 'conv4_3_bn.bias', '31.running_mean': 'conv4_3_bn.running_mean', '31.running_var': 'conv4_3_bn.running_var',
            '34.weight': 'conv5_1.weight',
            '35.weight': 'conv5_1_bn.weight', '35.bias': 'conv5_1_bn.bias', '35.running_mean': 'conv5_1_bn.running_mean', '35.running_var': 'conv5_1_bn.running_var',
            '37.weight': 'conv5_2.weight',
            '38.weight': 'conv5_2_bn.weight', '38.bias': 'conv5_2_bn.bias', '38.running_mean': 'conv5_2_bn.running_mean', '38.running_var': 'conv5_2_bn.running_var',
            '40.weight': 'conv5_3.weight',
            '41.weight': 'conv5_3_bn.weight', '41.bias': 'conv5_3_bn.bias', '41.running_mean': 'conv5_3_bn.running_mean', '41.running_var': 'conv5_3_bn.running_var'}
    if bias:
        layer_mappings.update({'0.bias': 'conv1_1.bias',
            '3.bias': 'conv1_2.bias',
            '7.bias': 'conv2_1.bias',
            '10.bias': 'conv2_2.bias',
            '14.bias': 'conv3_1.bias',
            '17.bias': 'conv3_2.bias',
            '20.bias': 'conv3_3.bias',
            '24.bias': 'conv4_1.bias',
            '27.bias': 'conv4_2.bias',
            '30.bias': 'conv4_3.bias',
            '34.bias': 'conv5_1.bias',
            '37.bias': 'conv5_2.bias',
            '40.bias': 'conv5_3.bias'
        })
    #Update according to generated mapping
    pretrained_dict = {layer_mappings[k]: v for k, v in vgg16_features_dict.items() if k in layer_mappings}
    return pretrained_dict


def map_state_dict_tf(vgg16_features, bias):
    pretrained_dict = {
        'conv1_1.weight': torch.from_numpy(vgg16_features['conv1_1'][0].transpose((3, 2, 0, 1))).float(),
        'conv1_2.weight': torch.from_numpy(vgg16_features['conv1_2'][0].transpose((3, 2, 0, 1))).float(),
        'conv2_1.weight': torch.from_numpy(vgg16_features['conv2_1'][0].transpose((3, 2, 0, 1))).float(),
        'conv2_2.weight': torch.from_numpy(vgg16_features['conv2_2'][0].transpose((3, 2, 0, 1))).float(),
        'conv3_1.weight': torch.from_numpy(vgg16_features['conv3_1'][0].transpose((3, 2, 0, 1))).float(),
        'conv3_2.weight': torch.from_numpy(vgg16_features['conv3_2'][0].transpose((3, 2, 0, 1))).float(),
        'conv3_3.weight': torch.from_numpy(vgg16_features['conv3_3'][0].transpose((3, 2, 0, 1))).float(),
        'conv4_1.weight': torch.from_numpy(vgg16_features['conv4_1'][0].transpose((3, 2, 0, 1))).float(),
        'conv4_2.weight': torch.from_numpy(vgg16_features['conv4_2'][0].transpose((3, 2, 0, 1))).float(),
        'conv4_3.weight': torch.from_numpy(vgg16_features['conv4_3'][0].transpose((3, 2, 0, 1))).float(),
        'conv5_1.weight': torch.from_numpy(vgg16_features['conv5_1'][0].transpose((3, 2, 0, 1))).float(),
        'conv5_2.weight': torch.from_numpy(vgg16_features['conv5_2'][0].transpose((3, 2, 0, 1))).float(),
        'conv5_3.weight': torch.from_numpy(vgg16_features['conv5_3'][0].transpose((3, 2, 0, 1))).float(),
    }
    if bias:
        pretrained_dict.update({
            'conv1_1.bias': torch.from_numpy(vgg16_features['conv1_1'][1]).float(),
            'conv1_2.bias': torch.from_numpy(vgg16_features['conv1_2'][1]).float(),
            'conv2_1.bias': torch.from_numpy(vgg16_features['conv2_1'][1]).float(),
            'conv2_2.bias': torch.from_numpy(vgg16_features['conv2_2'][1]).float(),
            'conv3_1.bias': torch.from_numpy(vgg16_features['conv3_1'][1]).float(),
            'conv3_2.bias': torch.from_numpy(vgg16_features['conv3_2'][1]).float(),
            'conv3_3.bias': torch.from_numpy(vgg16_features['conv3_3'][1]).float(),
            'conv4_1.bias': torch.from_numpy(vgg16_features['conv4_1'][1]).float(),
            'conv4_2.bias': torch.from_numpy(vgg16_features['conv4_2'][1]).float(),
            'conv4_3.bias': torch.from_numpy(vgg16_features['conv4_3'][1]).float(),
            'conv5_1.bias': torch.from_numpy(vgg16_features['conv5_1'][1]).float(),
            'conv5_2.bias': torch.from_numpy(vgg16_features['conv5_2'][1]).float(),
            'conv5_3.bias': torch.from_numpy(vgg16_features['conv5_3'][1]).float()
        })

    return pretrained_dict
