#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import shutil


def execute_build(args):
    """
    Create docker build command based on user input arguments
    and execute the command
    Args:
        args (argparse.Namespace): Build arguments
    """
    if not os.path.exists(args.docker_file):
        print('Dockerfile %s not found! Exiting' % args.docker_file)
        return

    cmd = 'DOCKER_BUILDKIT=1 docker build '
    cmd += '--rm --ssh default '
    cmd += '--network=host '
    if args.no_cache:
        cmd += '--no-cache '
    cmd += '-t %s -f %s .' % (args.image, args.docker_file)

    print('command = \n\n', cmd)

    if not args.dry_run:
        os.system(cmd)


if __name__ == '__main__':
    default_image_name = "diffhand"

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str,
                        default=default_image_name,
                        help='name for new docker image')

    parser.add_argument('--no-cache', action='store_true',
                        help='0 if should build without using cache')

    parser.add_argument('-f', '--docker-file', type=str,
                        default='Dockerfile',
                        help='which Dockerfile to build from')

    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='1 if we should only print the build command '
                             'without executing')

    args = parser.parse_args()
    execute_build(args)