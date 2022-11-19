# USAGE
# python main.py train --style ./path/to/style/image.jpg(video.mp4)   \
#                      --dataset ./path/to/dataset \
#                      --weights ./path/to/weights  \
#                      --batch 2

# python main.py evaluate --content ./path/to/content/image.jpg   \
#                         --weights ./path/to/weights \
#                         --result ./path/to/save/results/image.jpg

from distutils.cmd import Command
import os
import argparse
from .train import trainer
from .evaluate import transfer

CONTENT_WEIGHT = 6e0
STYLE_WEIGHT = 2e-3
TV_WEIGHT = 6e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 2

# STYLE_IMAGE = './images/style/udnie.jpg'
# CONTENT_IMAGE = './images/content/chicago.jpg'
# DATASET_PATH = '../datasets/train2014'
# WEIGHTS_PATH = './weights/wave/weights'
# RESULT_NAME = './images/results/Result.jpg'

STYLE_IMAGE = './media/images/style/udnie.jpg'
CONTENT_IMAGE = './media/images/content/chicago.jpg'
DATASET_PATH = './media/datasets/train2014'
WEIGHTS_PATH = './media/weights/wave/weights'
RESULT_NAME = './media/images/results/Result.jpg'


def main():
    # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Fast Style Transfer')
    # parser.add_argument('command',
    #                      metavar='<command>',
    #                      help="'train' or 'evaluate'")
    # parser.add_argument('--debug', required=False, type=bool,
    #                      metavar=False,
    #                      help='Whether to print the loss',
    #                      default=False)
    # parser.add_argument('--dataset', required=False,
    #                      metavar=DATASET_PATH,
    #                      default=DATASET_PATH)
    # parser.add_argument('--style', required=False,
    #                      metavar=STYLE_IMAGE,
    #                      help='Style image to train the specific style',
    #                      default=STYLE_IMAGE) 
    # parser.add_argument('--content', required=False,
    #                      metavar=CONTENT_IMAGE,
    #                      help='Content image/video to evaluate with',
    #                      default=CONTENT_IMAGE)  
    # parser.add_argument('--weights', required=False,
    #                      metavar=WEIGHTS_PATH,
    #                      help='Checkpoints directory',
    #                      default=WEIGHTS_PATH)
    # parser.add_argument('--result', required=False,
    #                      metavar=RESULT_NAME,
    #                      help='Path to the transfer results',
    #                      default=RESULT_NAME)
    # parser.add_argument('--batch', required=False, type=int,
    #                      metavar=BATCH_SIZE,
    #                      help='Training batch size',
    #                      default=BATCH_SIZE)
    # parser.add_argument('--max_dim', required=False, type=int,
    #                      metavar=None,
    #                      help='Resize the result image to desired size or remain as the original',
    #                      default=None)

    # args = parser.parse_args()

    command = "evaluate"
    debug = False
    style = STYLE_IMAGE
    dataset = DATASET_PATH
    content = CONTENT_IMAGE
    weights = WEIGHTS_PATH
    batch = BATCH_SIZE 
    result = RESULT_NAME
    max_dim = None

    # Validate arguments
    if command == "train":
        assert os.path.exists(dataset), 'dataset path not found !'
        assert os.path.exists(style), 'style image not found !'
        assert batch > 0
        assert NUM_EPOCHS > 0
        assert CONTENT_WEIGHT >= 0
        assert STYLE_WEIGHT >= 0
        assert TV_WEIGHT >= 0
        assert LEARNING_RATE >= 0

        parameters = {
                'style_file' : style,
                'dataset_path' : dataset,
                'weights_path' : weights,
                'debug' : debug,
                'content_weight' : CONTENT_WEIGHT,
                'style_weight' : STYLE_WEIGHT,
                'tv_weight' : TV_WEIGHT,
                'learning_rate' : LEARNING_RATE,
                'batch_size' : batch,
                'epochs' : NUM_EPOCHS,
            }

        trainer(**parameters)


    elif command == "evaluate":
        assert content, 'content image/video not found !'
        assert weights, 'weights path not found !'

        parameters = {
                'content' : content,
                'weights' : weights,
                'max_dim' : max_dim,
                'result' : result,
            }

        transfer(**parameters)


    else:
        print('Example usage : python main.py evaluate --content ./path/to/content/image.jpg')
        
        
if __name__ == '__main__':
    main()