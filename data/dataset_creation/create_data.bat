@echo off
setlocal

pushd "%~dp0..\.."

REM Refactor annotations FIRST: it produces data/splits/updated_annotations.csv,
REM which every image script below reads. It used to run last, so a fresh checkout
REM died on the first script with "updated_annotations.csv not found".
echo Refactoring annotations...
python data/splits/refactor_annotations.py

echo Creating full image dataset...
python data/dataset_creation/images/full_image_dataset.py
echo Creating skin only dataset...
python data/dataset_creation/images/skin_only_dataset.py
echo Creating skin parsed dataset (skin only, facial hair removed)...
python data/dataset_creation/images/skin_parsed_dataset.py
REM cheeks reads the full-image crops above, so it must come after full_image_dataset.
echo Creating cheeks and nose dataset...
python data/dataset_creation/images/cheeks_and_nose_dataset.py

echo All datasets created successfully.

echo Creating train/test splits...
python data/splits/images_train_test_splits.py
python data/splits/individual_train_test_splits.py

popd
endlocal