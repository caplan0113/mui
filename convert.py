import csv

def remove_y_columns(input_file, output_file):
    """
    *_y という名前の列をすべて削除して、新しいCSVとして出力する。

    :param input_file: 元のCSVファイルパス
    :param output_file: 出力するCSVファイルパス
    """
    with open(input_file, 'r', encoding='utf-8-sig') as f_in:
        reader = csv.DictReader(f_in)
        # ヘッダーから *_y を除外
        original_fieldnames = [field.strip() for field in reader.fieldnames]

        # 削除対象：_yで終わるが、_rot_yではない列
        filtered_fieldnames = [
            name for name in original_fieldnames
            if not (name.endswith('_y') and not name.endswith('_rot_y'))
        ]
        data = [row for row in reader]

    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=filtered_fieldnames)
        writer.writeheader()

        for row in data:
            # y列を除いた辞書を構築
            filtered_row = {k.strip(): v for k, v in row.items() if k.strip() in filtered_fieldnames}
            writer.writerow(filtered_row)

FROBOT = "data/{0:02d}/{0:02d}_{1:1d}_robot.csv"
for i in range(2, 5):
    for j in range(0, 5):
        if (i == 2 and j == 4):
            continue
        remove_y_columns(FROBOT.format(i, j), FROBOT.format(i, j))
