import os
import shutil
import random

def split_dataset(source_dir, output_dir, train=0.70, val=0.15, seed=42):
    """
    source_dir structure:
        raw_images/
            cats/
            dogs/

    output_dir structure created:
        dataset/
            train/cats/, train/dogs/
            val/cats/,   val/dogs/
            test/cats/,  test/dogs/
    """
    random.seed(seed)
    test = round(1.0 - train - val, 2)

    print(f"Split → Train: {int(train*100)}% | Val: {int(val*100)}% | Test: {int(test*100)}%\n")

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * train)
        n_val   = int(n * val)

        buckets = {
            'train': images[:n_train],
            'val':   images[n_train : n_train + n_val],
            'test':  images[n_train + n_val:]
        }

        for split_name, files in buckets.items():
            dest = os.path.join(output_dir, split_name, class_name)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(class_path, f), os.path.join(dest, f))

            print(f"  {split_name:6} / {class_name:10} → {len(files)} images")
        print()

    print("Done! Dataset ready at:", output_dir)


split_dataset(
    source_dir='catdog/',
    output_dir='dataset/',
    train=0.70,
    val=0.15
    # test gets the remaining 15% automatically
)