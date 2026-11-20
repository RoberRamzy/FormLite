"""Run dataset preparation, CUDA training, or the local video interface."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    preparation = subcommands.add_parser('prepare', help='Clean the videos and extract body features')
    preparation.add_argument('--manifest', type=Path, default=Path('exercise_recognition_4/manifest.csv'))
    preparation.add_argument('--jobs', type=int, default=8)
    preparation.add_argument('--scan-only', action='store_true', help='Run video and duplicate checks only')
    training = subcommands.add_parser('train', help='Train and evaluate on the CUDA GPU')
    training.add_argument('--epochs', type=int, default=50)
    serve = subcommands.add_parser('serve', help='Start the local video upload page')
    serve.add_argument('--port', type=int, default=8765)
    for subcommand in (preparation, training, serve):
        subcommand.add_argument('--output', type=Path, default=Path('exercise_motion_run'))
    for subcommand in (preparation, serve):
        subcommand.add_argument('--pose-model', type=Path,
                                default=Path('.exercise-motion-assets/pose_landmarker_lite.task'))
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            from .prepare import prepare

            if args.jobs < 1:
                parser.error('--jobs must be positive')
            if not args.scan_only and not args.pose_model.is_file():
                parser.error(f'Pose Landmarker model asset is missing: {args.pose_model}')
            prepare(args.manifest, args.pose_model, args.output, args.jobs, args.scan_only)
        elif args.command == 'train':
            from .classifier import train

            if not 1 <= args.epochs <= 50:
                parser.error('--epochs must be between 1 and 50')
            train(args.output, epochs=args.epochs)
        else:
            from .app import create_app

            create_app(args.output / 'model.pt', args.pose_model).run(
                host='127.0.0.1', port=args.port, debug=False, use_reloader=False,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
