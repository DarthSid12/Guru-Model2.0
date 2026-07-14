import os
import cv2

import numpy as np

ROOT = "data/thatcher_data/faces"

# Images you review
INSPECT_DIR = os.path.join(
    ROOT,
    "cnn",
    "upright",
    "thatcher"
)

VARIANTS = [
    ("cnn", "upright", "normal"),
    ("cnn", "upright", "thatcher"),
    ("cnn", "inverted", "normal"),
    ("cnn", "inverted", "thatcher"),
    ("lp", "upright", "normal"),
    ("lp", "upright", "thatcher"),
    ("lp", "inverted", "normal"),
    ("lp", "inverted", "thatcher"),
]


def delete_all_versions(identity, filename):

    deleted = 0

    for representation, orientation, condition in VARIANTS:

        path = os.path.join(
            ROOT,
            representation,
            orientation,
            condition,
            identity,
            filename,
        )

        if os.path.exists(path):
            os.remove(path)
            deleted += 1

    return deleted

def main():

    identities = sorted(os.listdir(INSPECT_DIR))

    for identity in identities:

        identity_dir = os.path.join(
            INSPECT_DIR,
            identity
        )

        if not os.path.isdir(identity_dir):
            continue

        images = sorted(os.listdir(identity_dir))

        for image_name in images:

            upright_path = os.path.join(
                ROOT,
                "cnn",
                "upright",
                "thatcher",
                identity,
                image_name
            )

            inverted_path = os.path.join(
                ROOT,
                "cnn",
                "inverted",
                "thatcher",
                identity,
                image_name
            )

            upright = cv2.imread(upright_path)
            inverted = cv2.imread(inverted_path)

            if upright is None or inverted is None:
                continue

            # Just in case the sizes differ
            if upright.shape != inverted.shape:
                inverted = cv2.resize(
                    inverted,
                    (upright.shape[1], upright.shape[0])
                )

            display = np.hstack((upright, inverted))

            cv2.imshow(
                f"ID: {identity}   IMG: {image_name}   Left=Upright  Right=Inverted    y=keep  n=delete  q=quit",
                display
            )

            while True:

                key = cv2.waitKey(0) & 0xFF

                if key == ord("y"):
                    break

                elif key == ord("n"):

                    n = delete_all_versions(
                        identity,
                        image_name
                    )

                    print(
                        f"Deleted {n} files for "
                        f"{identity}/{image_name}"
                    )
                    break

                elif key == ord("q"):

                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()

def main_old():

    identities = sorted(os.listdir(INSPECT_DIR))

    for identity in identities:

        identity_dir = os.path.join(
            INSPECT_DIR,
            identity
        )

        if not os.path.isdir(identity_dir):
            continue

        images = sorted(os.listdir(identity_dir))

        for image_name in images:

            path = os.path.join(identity_dir, image_name)

            img = cv2.imread(path)

            if img is None:
                continue

            cv2.imshow(
                f"ID: {identity} IM: {image}    y = keep    n = delete all versions    q = quit",
                img
            )

            while True:

                key = cv2.waitKey(0) & 0xFF

                if key == ord("y"):
                    break

                elif key == ord("n"):

                    n = delete_all_versions(
                        identity,
                        image_name
                    )

                    print(
                        f"Deleted {n} files for "
                        f"{identity}/{image_name}"
                    )
                    break

                elif key == ord("q"):

                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()