from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io, uuid, zipfile, json
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

app = Flask(__name__)
CORS(app, origins="*")


@app.route('/api/default-clean', methods=['POST'])
def auto_clean():
    df = None
    if 'file' in request.files:
        file = request.files['file']
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
    elif 'data_url' in request.args:
        try:
            df = pd.read_csv(request.args['data_url'])
        except Exception as e:
            return jsonify({'error': f'Failed to fetch dataset from URL: {str(e)}'}), 400
    else:
        return jsonify({'error': 'No file or data_url provided'}), 400

    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown", inplace=True)

    cleaned_df = df
    try:
        profile = ProfileReport(cleaned_df, title="Post-Cleaning EDA Report", minimal=False, explorative=True)
        eda_html_str = profile.to_html()
    except Exception as e:
        print(f"[EDA fallback]: {e}")
        eda_html_str = "<html><body>EDA generation failed.</body></html>"

    csv_buffer = io.StringIO()
    cleaned_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('cleaned_dataset.csv', csv_buffer.getvalue())
        zf.writestr('eda_report.html', eda_html_str)
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'auto_cleaned_result_{uuid.uuid4().hex}.zip'
    )


@app.route('/api/manual-clean', methods=['POST'])
def manual_clean():
    df = None

    try:
        config = json.loads(request.form.get('config'))
    except Exception as e:
        return jsonify({'error': f'Invalid config JSON: {str(e)}'}), 400

    if 'file' in request.files:
        file = request.files['file']
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
    elif 'url' in request.args:
        try:
            df = pd.read_csv(request.args.get('url'))
        except Exception as e:
            return jsonify({'error': f'Failed to load dataset from URL: {str(e)}'}), 400
    else:
        return jsonify({'error': 'No file or url provided'}), 400

    df = df.copy()

    # Drop columns
    for col in config.get('columns_to_drop', []):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Handle missing values
    for col, strategy in config.get('imputation', {}).items():
        if col not in df.columns:
            continue
        if strategy == 'remove':
            df = df[df[col].notnull()]
        elif strategy == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
        elif strategy == 'constant':
            df[col].fillna("Unknown", inplace=True)

    # Handle outliers
    if config.get('outlier_handling', {}).get('method') == 'iqr':
        for col in df.select_dtypes(include='number').columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Encoding
    for col, method in config.get('encoding', {}).items():
        if col not in df.columns:
            continue
        if method == 'onehot':
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            reshaped = df[[col]].astype(str)
            encoded = ohe.fit_transform(reshaped)
            encoded_cols = ohe.get_feature_names_out([col])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df.drop(columns=[col], inplace=True)
            df = pd.concat([df, encoded_df], axis=1)
        elif method == 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Scaling
    for col, method in config.get('scaling', {}).items():
        if col not in df.columns:
            continue
        if method == 'standard':
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])

    try:
        profile = ProfileReport(df, title="Post-Manual-Cleaning EDA", minimal=False, explorative=True)
        eda_html_str = profile.to_html()
    except Exception as e:
        print(f"[EDA fallback]: {e}")
        eda_html_str = "<html><body>EDA generation failed.</body></html>"

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('cleaned_dataset.csv', csv_buffer.getvalue())
        zf.writestr('eda_report.html', eda_html_str)
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'manual_cleaned_result_{uuid.uuid4().hex}.zip'
    )


if __name__ == "__main__":
    app.run(debug=True)
