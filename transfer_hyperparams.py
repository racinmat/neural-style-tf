import itertools
import tensorflow as tf
import time
import os.path as osp
import neural_style
from datetime import datetime
from neural_style import parse_args, get_init_image, build_optimizer, get_content_image, get_style_images, \
    build_model, compute_style_transfer, write_image_output


def main():
    neural_style.args = parse_args()
    args = neural_style.args
    hparams = {
        # 'style_weight': [1e2, 1e3, 1e4, 1e5, 1e6],
        # 'content_weight': [5e-2, 5e-1, 5e0, 5e1, 5e2],
        'style_weight': [1e3, 1e4, 1e5],
        'content_weight': [5e-1, 5e0, 5e1],
        'init_img_type': ['random', 'content', 'style'],
        # 'original_colors': [True, False],  # tried in hparams-result-2019-03-28--21-36, better with color transfer
    }
    # constraints must hold for config
    constraints = [
        lambda p: (p['style_weight'] / p['content_weight'] >= 1) and (p['style_weight'] / p['content_weight'] <= 500),    # empirically found that too high loss compared to content deforms too much, and too low does not change image
    ]
    out_dir = osp.join(args.img_output_dir, 'hparams-result-' + datetime.now().strftime('%Y-%m-%d--%H-%M'))
    with tf.Graph().as_default(), tf.device(args.device), tf.Session() as sess:
        content_img = get_content_image(args.content_img)
        style_imgs = get_style_images(content_img)
        net = build_model(content_img, args.verbose, args.model_weights)

        for param_set in itertools.product(*hparams.values()):
            params_dict = dict(zip(hparams.keys(), param_set))
            if any([not c(params_dict) for c in constraints]):
                continue    # skip if any of constrains does not hols

            for key, value in params_dict.items():
                setattr(neural_style.args, key, value)

            print('\n---- RENDERING SINGLE IMAGE ----\n')
            init_img = get_init_image(neural_style.args.init_img_type, content_img, style_imgs)

            tick = time.time()
            L_total, net, optimizer = build_optimizer(net, content_img, init_img, sess, style_imgs)

            output_img = compute_style_transfer(L_total, content_img, init_img, net, optimizer, sess)
            tock = time.time()
            print('Single image elapsed time: {}'.format(tock - tick))

            write_image_output(out_dir, output_img, content_img, style_imgs, init_img, prefix=str(param_set)+'_')


if __name__ == '__main__':
    main()
