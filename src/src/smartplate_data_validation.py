# ============================================================
# SMARTPLATE - COMPLETE DATA VALIDATION
# ============================================================

# YOUR ACTUAL DATASET PATH
INPUT_FILE = r"C:\Users\Aaditya babaso patil\Desktop\SmartPlate_Data_Mining_Project\data\smartplate_cafeteria_dataset.csv"


def validate_dataset(file_path):

    print("=" * 70)
    print("SMARTPLATE - DATA VALIDATION")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    try:

        df = pd.read_csv(file_path)

        print("\n Dataset loaded successfully.")

    except FileNotFoundError:

        print("\n ERROR: Dataset file not found.")
        print("Check the file path:")
        print(file_path)

        return None

    # ========================================================
    # 2. DATASET SHAPE
    # ========================================================

    print("\n" + "=" * 70)
    print("1. DATASET SHAPE")
    print("=" * 70)

    print("Number of Rows    :", df.shape[0])
    print("Number of Columns :", df.shape[1])

    # ========================================================
    # 3. COLUMN NAMES
    # ========================================================

    print("\n" + "=" * 70)
    print("2. COLUMN NAMES")
    print("=" * 70)

    for number, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{number:02d}. {column}"
        )

    # ========================================================
    # 4. FIRST 5 ROWS
    # ========================================================

    print("\n" + "=" * 70)
    print("3. FIRST 5 ROWS")
    print("=" * 70)

    print(
        df.head()
    )

    # ========================================================
    # 5. DATA TYPES
    # ========================================================

    print("\n" + "=" * 70)
    print("4. DATA TYPES")
    print("=" * 70)

    print(
        df.dtypes
    )

    # ========================================================
    # 6. MISSING VALUES
    # ========================================================

    print("\n" + "=" * 70)
    print("5. MISSING VALUES")
    print("=" * 70)

    missing_values = df.isnull().sum()

    print(
        missing_values
    )

    total_missing = (
        missing_values.sum()
    )

    print(
        "\nTotal Missing Values:",
        total_missing
    )

    if total_missing == 0:

        print(
            " PASS: No missing values."
        )

    else:

        print(
            "⚠️ WARNING: Missing values found."
        )

    # ========================================================
    # 7. DUPLICATE ROW