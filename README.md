# Decision Tree Regressor - BigMart Sales Prediction

This project uses a Decision Tree Regressor to predict `Item_Outlet_Sales` from the BigMart sales dataset.

## Files

- `Decision Tree Regressor.ipynb` - Jupyter notebook containing the full data loading, preprocessing, training, and evaluation workflow.
- `Train.csv` - Training dataset used by the notebook.
- `Test.csv` - Optional test dataset (not currently used by the notebook).
- `requirements.txt` - Python dependencies required to run the notebook.

## Requirements

The notebook requires the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open `Decision Tree Regressor.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the notebook cells sequentially.

## Notes

- The notebook reads `Train.csv` from the project root.
- Categorical columns are encoded using `LabelEncoder`.
- The model is evaluated using `R2`, `MAE`, `MSE`, and `RMSE`.
