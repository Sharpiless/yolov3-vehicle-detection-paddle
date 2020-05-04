import cv2
from paddle_model_cls.model_with_code.model import x2paddle_net

import argparse
import functools
import numpy as np
import paddle.fluid as fluid

class CarClassifier(object):
    
    def __init__(self):

        self.color_attrs = ['Black', 'Blue', 'Brown',
                    'Gray', 'Green', 'Pink',
                    'Red', 'White', 'Yellow']  # 车体颜色

        self.direction_attrs = ['Front', 'Rear']  # 拍摄位置

        self.type_attrs = ['passengerCar', 'saloonCar',
                    'shopTruck', 'suv', 'trailer', 'truck', 'van', 'waggon']  # 车辆类型

        self.init_params()


    # 定义一个预处理图像的函数
    def process_img(self, img, image_shape=[3, 224, 224]):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img = cv2.resize(img, (image_shape[1], image_shape[2]))
        #img = cv2.resize(img,(256,256))
        #img = crop_image(img, image_shape[1], True)

        # RBG img [224,224,3]->[3,224,224]
        img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
        #img = img.astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std

        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        return img


    def inference(self, img):
        fetch_list = [self.out.name]

        output = self.exe.run(self.eval_program,
                        fetch_list=fetch_list,
                        feed={'image': img})
        color_idx, direction_idx, type_idx = self.get_predict(np.array(output))

        color_name = self.color_attrs[color_idx]
        direction_name = self.direction_attrs[direction_idx]
        type_name = self.type_attrs[type_idx]

        return color_name, direction_name, type_name


    def get_predict(self, output):
        output = np.squeeze(output)
        pred_color = output[:9]
        pred_direction = output[9:11]
        pred_type = output[11:]

        color_idx = np.argmax(pred_color)
        direction_idx = np.argmax(pred_direction)
        type_idx = np.argmax(pred_type)

        return color_idx, direction_idx, type_idx

    def init_params(self):
        use_gpu = True
        # Attack graph
        adv_program = fluid.Program()

        # 完成初始化
        with fluid.program_guard(adv_program):
            input_layer = fluid.layers.data(
                name='image', shape=[3, 224, 224], dtype='float32')
            # 设置为可以计算梯度
            input_layer.stop_gradient = False

            # model definition
            _, out_logits = x2paddle_net(inputs=input_layer)
            self.out = fluid.layers.softmax(out_logits[0])

            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            self.exe = fluid.Executor(place)
            self.exe.run(fluid.default_startup_program())

            # 记载模型参数
            fluid.io.load_persistables(self.exe, './paddle_model_cls/model_with_code/')

        # 创建测试用评估模式
        self.eval_program = adv_program.clone(for_test=True)

    def predict(self, im):

        im_input = self.process_img(im)

        color_name, direction_name, type_name = self.inference(im_input)

        label = '颜色：{}\n朝向：{}\n类型：{}'.format(color_name, direction_name, type_name)

        return label


if __name__ == '__main__':
    
    net = CarClassifier()
    # im_pt = './a.jpg'
    im_pt = './a.png'
    img = cv2.imread(im_pt)
    label = net.predict(img)
    print(label)
    img = cv2.imread(im_pt)


    cv2.imshow('a', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
