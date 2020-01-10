# Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection


```python
import numpy as np 
import pandas as pd
import torch
from data_loader import *
from main import *
from tqdm import tqdm
```

## KDD Cup 1999 Data (10% subset)
This is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between "bad" connections, called intrusions or attacks, and "good" normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment. 


```python
transaction_data = pd.read_csv("ieee-fraud-detection/train_transaction.csv", header=0)
transaction_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>...</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2987000</td>
      <td>0</td>
      <td>86400</td>
      <td>68.50</td>
      <td>W</td>
      <td>13926</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>discover</td>
      <td>142.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2987001</td>
      <td>0</td>
      <td>86401</td>
      <td>29.00</td>
      <td>W</td>
      <td>2755</td>
      <td>404.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2987002</td>
      <td>0</td>
      <td>86469</td>
      <td>59.00</td>
      <td>W</td>
      <td>4663</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2987003</td>
      <td>0</td>
      <td>86499</td>
      <td>50.00</td>
      <td>W</td>
      <td>18132</td>
      <td>567.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>117.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2987004</td>
      <td>0</td>
      <td>86506</td>
      <td>50.00</td>
      <td>H</td>
      <td>4497</td>
      <td>514.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>590535</td>
      <td>3577535</td>
      <td>0</td>
      <td>15811047</td>
      <td>49.00</td>
      <td>W</td>
      <td>6550</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>visa</td>
      <td>226.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>590536</td>
      <td>3577536</td>
      <td>0</td>
      <td>15811049</td>
      <td>39.50</td>
      <td>W</td>
      <td>10444</td>
      <td>225.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>590537</td>
      <td>3577537</td>
      <td>0</td>
      <td>15811079</td>
      <td>30.95</td>
      <td>W</td>
      <td>12037</td>
      <td>595.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>590538</td>
      <td>3577538</td>
      <td>0</td>
      <td>15811088</td>
      <td>117.00</td>
      <td>W</td>
      <td>7826</td>
      <td>481.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>590539</td>
      <td>3577539</td>
      <td>0</td>
      <td>15811131</td>
      <td>279.95</td>
      <td>W</td>
      <td>15066</td>
      <td>170.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>590540 rows × 394 columns</p>
</div>




```python
identity_data = pd.read_csv("ieee-fraud-detection/train_identity.csv", header=0)
identity_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>...</th>
      <th>id_31</th>
      <th>id_32</th>
      <th>id_33</th>
      <th>id_34</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2987004</td>
      <td>0.0</td>
      <td>70787.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>samsung browser 6.2</td>
      <td>32.0</td>
      <td>2220x1080</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>mobile</td>
      <td>SAMSUNG SM-G892A Build/NRD90M</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2987008</td>
      <td>-5.0</td>
      <td>98945.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>mobile safari 11.0</td>
      <td>32.0</td>
      <td>1334x750</td>
      <td>match_status:1</td>
      <td>T</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>mobile</td>
      <td>iOS Device</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2987010</td>
      <td>-5.0</td>
      <td>191631.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>Windows</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2987011</td>
      <td>-5.0</td>
      <td>221832.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2987016</td>
      <td>0.0</td>
      <td>7460.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>chrome 62.0</td>
      <td>24.0</td>
      <td>1280x800</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>MacOS</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>144228</td>
      <td>3577521</td>
      <td>-15.0</td>
      <td>145955.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>chrome 66.0 for android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>F</td>
      <td>mobile</td>
      <td>F3111 Build/33.3.A.1.97</td>
    </tr>
    <tr>
      <td>144229</td>
      <td>3577526</td>
      <td>-5.0</td>
      <td>172059.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>-5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>chrome 55.0 for android</td>
      <td>32.0</td>
      <td>855x480</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>F</td>
      <td>mobile</td>
      <td>A574BL Build/NMF26F</td>
    </tr>
    <tr>
      <td>144230</td>
      <td>3577529</td>
      <td>-20.0</td>
      <td>632381.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>-36.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>chrome 65.0 for android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>F</td>
      <td>mobile</td>
      <td>Moto E (4) Plus Build/NMA26.42-152</td>
    </tr>
    <tr>
      <td>144231</td>
      <td>3577531</td>
      <td>-5.0</td>
      <td>55528.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>chrome 66.0</td>
      <td>24.0</td>
      <td>2560x1600</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>F</td>
      <td>desktop</td>
      <td>MacOS</td>
    </tr>
    <tr>
      <td>144232</td>
      <td>3577534</td>
      <td>-45.0</td>
      <td>339406.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-10.0</td>
      <td>-100.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>chrome 66.0 for android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>F</td>
      <td>mobile</td>
      <td>RNE-L03 Build/HUAWEIRNE-L03</td>
    </tr>
  </tbody>
</table>
<p>144233 rows × 41 columns</p>
</div>




```python
data = transaction_data.set_index('TransactionID').join(identity_data.set_index('TransactionID'))
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>...</th>
      <th>id_31</th>
      <th>id_32</th>
      <th>id_33</th>
      <th>id_34</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2987000</td>
      <td>0</td>
      <td>86400</td>
      <td>68.50</td>
      <td>W</td>
      <td>13926</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>discover</td>
      <td>142.0</td>
      <td>credit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987001</td>
      <td>0</td>
      <td>86401</td>
      <td>29.00</td>
      <td>W</td>
      <td>2755</td>
      <td>404.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987002</td>
      <td>0</td>
      <td>86469</td>
      <td>59.00</td>
      <td>W</td>
      <td>4663</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>debit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987003</td>
      <td>0</td>
      <td>86499</td>
      <td>50.00</td>
      <td>W</td>
      <td>18132</td>
      <td>567.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>117.0</td>
      <td>debit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987004</td>
      <td>0</td>
      <td>86506</td>
      <td>50.00</td>
      <td>H</td>
      <td>4497</td>
      <td>514.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>...</td>
      <td>samsung browser 6.2</td>
      <td>32.0</td>
      <td>2220x1080</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>mobile</td>
      <td>SAMSUNG SM-G892A Build/NRD90M</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3577535</td>
      <td>0</td>
      <td>15811047</td>
      <td>49.00</td>
      <td>W</td>
      <td>6550</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>visa</td>
      <td>226.0</td>
      <td>debit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3577536</td>
      <td>0</td>
      <td>15811049</td>
      <td>39.50</td>
      <td>W</td>
      <td>10444</td>
      <td>225.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>debit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3577537</td>
      <td>0</td>
      <td>15811079</td>
      <td>30.95</td>
      <td>W</td>
      <td>12037</td>
      <td>595.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>debit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3577538</td>
      <td>0</td>
      <td>15811088</td>
      <td>117.00</td>
      <td>W</td>
      <td>7826</td>
      <td>481.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>debit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3577539</td>
      <td>0</td>
      <td>15811131</td>
      <td>279.95</td>
      <td>W</td>
      <td>15066</td>
      <td>170.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>590540 rows × 433 columns</p>
</div>



### Pre-processing
"isFraud" = 0 -> normal, "isFraud" = 1 -> anomaly. 

Next, the categorical variables are converted to a one hot encoding representation. My implementation is a bit different from the original paper in this aspect. Since I am only using the 10% subset to generate the columns, I get 118 features instead of 120 as reported in the paper.


```python
one_hot_ProductCD = pd.get_dummies(data["ProductCD"])
one_hot_card4 = pd.get_dummies(data["card4"])
one_hot_card6 = pd.get_dummies(data["card6"])
one_hot_Pemaildomain = pd.get_dummies(data["P_emaildomain"])
one_hot_Remaildomain = pd.get_dummies(data["R_emaildomain"])
one_hot_M1 = pd.get_dummies(data["M1"])
one_hot_M2 = pd.get_dummies(data["M2"])
one_hot_M3 = pd.get_dummies(data["M3"])
one_hot_M4 = pd.get_dummies(data["M4"])
one_hot_M5 = pd.get_dummies(data["M5"])
one_hot_M6 = pd.get_dummies(data["M6"])
one_hot_M7 = pd.get_dummies(data["M7"])
one_hot_M8 = pd.get_dummies(data["M8"])
one_hot_M9 = pd.get_dummies(data["M9"])
one_hot_id12 = pd.get_dummies(data["id_12"])
one_hot_id15 = pd.get_dummies(data["id_15"])
one_hot_id16 = pd.get_dummies(data["id_16"])
one_hot_id23 = pd.get_dummies(data["id_23"])
one_hot_id27 = pd.get_dummies(data["id_27"])
one_hot_id28 = pd.get_dummies(data["id_28"])
one_hot_id29 = pd.get_dummies(data["id_29"])
one_hot_id30 = pd.get_dummies(data["id_30"])
one_hot_id31 = pd.get_dummies(data["id_31"])
one_hot_id33 = pd.get_dummies(data["id_33"])
one_hot_id34 = pd.get_dummies(data["id_34"])
one_hot_id35 = pd.get_dummies(data["id_35"])
one_hot_id36 = pd.get_dummies(data["id_36"])
one_hot_id37 = pd.get_dummies(data["id_37"])
one_hot_id38 = pd.get_dummies(data["id_38"])
one_hot_DeviceType = pd.get_dummies(data["DeviceType"])
one_hot_DeviceInfo = pd.get_dummies(data["DeviceInfo"])
```


```python
data = data.drop("ProductCD",axis=1)
data = data.drop("card4",axis=1)
data = data.drop("card6",axis=1)
data = data.drop("P_emaildomain",axis=1)
data = data.drop("R_emaildomain",axis=1)
data = data.drop("M1",axis=1)
data = data.drop("M2",axis=1)
data = data.drop("M3",axis=1)
data = data.drop("M4",axis=1)
data = data.drop("M5",axis=1)
data = data.drop("M6",axis=1)
data = data.drop("M7",axis=1)
data = data.drop("M8",axis=1)
data = data.drop("M9",axis=1)
data = data.drop("id_12",axis=1)
data = data.drop("id_15",axis=1)
data = data.drop("id_16",axis=1)
data = data.drop("id_23",axis=1)
data = data.drop("id_27",axis=1)
data = data.drop("id_28",axis=1)
data = data.drop("id_29",axis=1)
data = data.drop("id_30",axis=1)
data = data.drop("id_31",axis=1)
data = data.drop("id_33",axis=1)
data = data.drop("id_34",axis=1)
data = data.drop("id_35",axis=1)
data = data.drop("id_36",axis=1)
data = data.drop("id_37",axis=1)
data = data.drop("id_38",axis=1)
data = data.drop("DeviceType",axis=1)
data = data.drop("DeviceInfo",axis=1)
```


```python
data = pd.concat([one_hot_ProductCD, one_hot_card4, one_hot_card6, 
                  one_hot_Pemaildomain, one_hot_Remaildomain, one_hot_M1, 
                  one_hot_M2, one_hot_M3, one_hot_M4, 
                  one_hot_M5, one_hot_M6, one_hot_M7, 
                  one_hot_M8, one_hot_M9, one_hot_id12, 
                  one_hot_id15, one_hot_id16, one_hot_id23, 
                  one_hot_id27, one_hot_id28, one_hot_id29, 
                  one_hot_id30, one_hot_id31, one_hot_id33, 
                  one_hot_id34, one_hot_id35, one_hot_id36, 
                  one_hot_id37, one_hot_id38, one_hot_DeviceType, 
                  one_hot_DeviceInfo, data],axis=1)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>H</th>
      <th>R</th>
      <th>S</th>
      <th>W</th>
      <th>american express</th>
      <th>discover</th>
      <th>mastercard</th>
      <th>visa</th>
      <th>charge card</th>
      <th>...</th>
      <th>id_17</th>
      <th>id_18</th>
      <th>id_19</th>
      <th>id_20</th>
      <th>id_21</th>
      <th>id_22</th>
      <th>id_24</th>
      <th>id_25</th>
      <th>id_26</th>
      <th>id_32</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2987000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987001</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987002</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2987004</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>542.0</td>
      <td>144.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2834 columns</p>
</div>




```python
#data.loc[:,"SAMSUNG SM-G892A Build/NRD90M"]
```




    TransactionID
    2987000    0
    2987001    0
    2987002    0
    2987003    0
    2987004    1
              ..
    3577535    0
    3577536    0
    3577537    0
    3577538    0
    3577539    0
    Name: SAMSUNG SM-G892A Build/NRD90M, Length: 590540, dtype: uint8




```python
proportions = data["isFraud"].value_counts()
print(proportions)
print("Anomaly Percentage",proportions[1] / proportions.sum())
```

    0    569877
    1     20663
    Name: isFraud, dtype: int64
    Anomaly Percentage 0.03499000914417313
    


```python
#proportions_alfa = data["type"].value_counts(normalize=True)
#print(proportions_alfa)
```

Normalize all the numeric variables.


```python
cols_to_norm = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", 
            "hot", "num_failed_logins", "num_compromised", "num_root", 
            "num_file_creations", "num_shells", "num_access_files", "count", "srv_count", 
            "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
            "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate" ]

#data.loc[:, cols_to_norm] = (data[cols_to_norm] - data[cols_to_norm].mean()) / data[cols_to_norm].std()
min_cols = data.loc[data["type"]==0 , cols_to_norm].min()
max_cols = data.loc[data["type"]==0 , cols_to_norm].max()

data.loc[:, cols_to_norm] = (data[cols_to_norm] - min_cols) / (max_cols - min_cols)
```

I saved the preprocessed data into a numpy file format and load it using the pytorch data loader.


```python
np.savez_compressed("kdd_cup",kdd=data.as_matrix())
```

    C:\Users\cncluser\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    

I initially implemented this to be ran in the command line and use argparse to get the hyperparameters. To make it runnable in a jupyter notebook, I had to create a dummy class for the hyperparameters.


```python
class hyperparams():
    def __init__(self, config):
        self.__dict__.update(**config)
defaults = {
    'lr' : 1e-4,
    'num_epochs' : 200,
    'batch_size' : 1024,
    'gmm_k' : 4,
    'lambda_energy' : 0.1,
    'lambda_cov_diag' : 0.005,
    'pretrained_model' : None,
    'mode' : 'train',
    'use_tensorboard' : False,
    'data_path' : 'kdd_cup.npz',

    'log_path' : './dagmm/logs',
    'model_save_path' : './dagmm/models',
    'sample_path' : './dagmm/samples',
    'test_sample_path' : './dagmm/test_samples',
    'result_path' : './dagmm/results',

    'log_step' : 194//4,
    'sample_step' : 194,
    'model_save_step' : 194,
}
```


```python
solver = main(hyperparams(defaults))
accuracy, precision, recall, f_score = solver.test()
```

    Elapsed 0:00:13.076672/0:00:07.943268 -- 0:34:43.567725 , Epoch [2/200], Iter [48/194], lr 0.0001, total_loss: 0.1430, sample_energy: -2.4255, recon_error: 0.0624, cov_diag: 64.6240
    


![png](output_20_1.png)


    phi tensor([0.2356, 0.1980, 0.3282, 0.2383]) mu tensor([[-0.5328,  0.8382,  0.5090],
            [-0.4987,  0.8608,  0.4772],
            [-0.4941,  0.8639,  0.4729],
            [-0.5665,  0.8157,  0.5406]]) cov tensor([[[ 0.2696,  0.1789, -0.2516],
             [ 0.1789,  0.1193, -0.1676],
             [-0.2516, -0.1676,  0.2355]],
    
            [[ 0.2868,  0.1903, -0.2676],
             [ 0.1903,  0.1268, -0.1781],
             [-0.2676, -0.1781,  0.2504]],
    
            [[ 0.2885,  0.1915, -0.2693],
             [ 0.1915,  0.1275, -0.1792],
             [-0.2693, -0.1792,  0.2519]],
    
            [[ 0.2497,  0.1658, -0.2331],
             [ 0.1658,  0.1105, -0.1553],
             [-0.2331, -0.1553,  0.2182]]])
    

     29%|███████████████████████▊                                                         | 57/194 [00:03<00:07, 17.44it/s]
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-10-db7e643ba5ba> in <module>
    ----> 1 solver = main(hyperparams(defaults))
          2 accuracy, precision, recall, f_score = solver.test()
    

    ~\dagmm\main.py in main(config)
         23 
         24     if config.mode == 'train':
    ---> 25         solver.train()
         26     elif config.mode == 'test':
         27         solver.test()
    

    ~\dagmm\solver.py in train(self)
         95                 input_data = self.to_var(input_data)
         96 
    ---> 97                 total_loss,sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
         98                 # Logging
         99                 loss = {}
    

    ~\dagmm\solver.py in dagmm_step(self, input_data)
        162         enc, dec, z, gamma = self.dagmm(input_data)
        163 
    --> 164         total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)
        165 
        166         self.reset_grad()
    

    ~\dagmm\model.py in loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag)
        166         phi, mu, cov = self.compute_gmm_params(z, gamma)
        167 
    --> 168         sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        169 
        170         loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
    

    ~\dagmm\model.py in compute_energy(self, z, phi, mu, cov, size_average)
        121         k, D, _ = cov.size()
        122 
    --> 123         z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        124 
        125         cov_inverse = []
    

    KeyboardInterrupt: 


### I copy pasted the testing code here in the notebook so we could play around the results.

### Incrementally compute for the GMM parameters across all training data for a better estimate


```python
solver.data_loader.dataset.mode="train"
solver.dagmm.eval()
N = 0
mu_sum = 0
cov_sum = 0
gamma_sum = 0

for it, (input_data, labels) in enumerate(solver.data_loader):
    input_data = solver.to_var(input_data)
    enc, dec, z, gamma = solver.dagmm(input_data)
    phi, mu, cov = solver.dagmm.compute_gmm_params(z, gamma)
    
    batch_gamma_sum = torch.sum(gamma, dim=0)
    
    gamma_sum += batch_gamma_sum
    mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
    cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
    
    N += input_data.size(0)
    
train_phi = gamma_sum / N
train_mu = mu_sum / gamma_sum.unsqueeze(-1)
train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

print("N:",N)
print("phi :\n",train_phi)
print("mu :\n",train_mu)
print("cov :\n",train_cov)
```


```python
train_energy = []
train_labels = []
train_z = []
for it, (input_data, labels) in enumerate(solver.data_loader):
    input_data = solver.to_var(input_data)
    enc, dec, z, gamma = solver.dagmm(input_data)
    sample_energy, cov_diag = solver.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
    
    train_energy.append(sample_energy.data.cpu().numpy())
    train_z.append(z.data.cpu().numpy())
    train_labels.append(labels.numpy())


train_energy = np.concatenate(train_energy,axis=0)
train_z = np.concatenate(train_z,axis=0)
train_labels = np.concatenate(train_labels,axis=0)
```

### Compute the energy of every sample in the test data


```python
solver.data_loader.dataset.mode="test"
test_energy = []
test_labels = []
test_z = []
for it, (input_data, labels) in enumerate(solver.data_loader):
    input_data = solver.to_var(input_data)
    enc, dec, z, gamma = solver.dagmm(input_data)
    sample_energy, cov_diag = solver.dagmm.compute_energy(z, size_average=False)
    test_energy.append(sample_energy.data.cpu().numpy())
    test_z.append(z.data.cpu().numpy())
    test_labels.append(labels.numpy())


test_energy = np.concatenate(test_energy,axis=0)
test_z = np.concatenate(test_z,axis=0)
test_labels = np.concatenate(test_labels,axis=0)
```


```python
combined_energy = np.concatenate([train_energy, test_energy], axis=0)
combined_z = np.concatenate([train_z, test_z], axis=0)
combined_labels = np.concatenate([train_labels, test_labels], axis=0)
```

### Compute for the threshold energy. Following the paper I just get the highest 20% and treat it as an anomaly. That corresponds to setting the threshold at the 80th percentile.


```python
thresh = np.percentile(combined_energy, 100 - 20)
print("Threshold :", thresh)
```


```python
pred = (test_energy>thresh).astype(int)
gt = test_labels.astype(int)
```


```python
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
```


```python
accuracy = accuracy_score(gt,pred)
precision, recall, f_score, support = prf(gt, pred, average='binary')
```


```python
print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy,precision, recall, f_score))
```

## Visualizing the z space
It's a little different from the paper's figure but I assume that's because of the small changes in my implementation.


```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib notebook
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_z[:,1],test_z[:,0], test_z[:,2], c=test_labels.astype(int))
ax.set_xlabel('Encoded')
ax.set_ylabel('Euclidean')
ax.set_zlabel('Cosine')
plt.show()
```


```python

```
