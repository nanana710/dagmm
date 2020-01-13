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
#WIndows
#transaction_data = pd.read_csv("C:/Users/cncluser/Downloads/ieee-fraud-detection/train_transaction.csv", header=0)
#macOS
transaction_data = pd.read_csv("/Users/nami/Downloads/ieee-fraud-detection/train_transaction.csv", header=0)

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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>...</th>
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
      <th>590535</th>
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
      <th>590536</th>
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
      <th>590537</th>
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
      <th>590538</th>
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
      <th>590539</th>
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
#Windows
#identity_data = pd.read_csv("C:/Users/cncluser/Downloads/ieee-fraud-detection/train_identity.csv", header=0)
#macOS
identity_data = pd.read_csv("/Users/nami/Downloads/ieee-fraud-detection/train_identity.csv", header=0)

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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>...</th>
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
      <th>144228</th>
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
      <th>144229</th>
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
      <th>144230</th>
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
      <th>144231</th>
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
      <th>144232</th>
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
      <th>2987000</th>
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
      <th>2987001</th>
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
      <th>2987002</th>
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
      <th>2987003</th>
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
      <th>2987004</th>
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
      <th>...</th>
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
      <th>3577535</th>
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
      <th>3577536</th>
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
      <th>3577537</th>
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
      <th>3577538</th>
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
      <th>3577539</th>
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
data_header = data.columns
data_header = data_header.drop("isFraud")
#data_header
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
      <th>2987000</th>
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
      <th>2987001</th>
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
      <th>2987002</th>
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
      <th>2987003</th>
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
      <th>2987004</th>
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
#proportions_alfa = data["isFraud"].value_counts(normalize=True)
#print(proportions_alfa)
```

Normalize all the numeric variables.


```python
cols_to_norm = data_header
print(cols_to_norm)

#data.loc[:, cols_to_norm] = (data[cols_to_norm] - data[cols_to_norm].mean()) / data[cols_to_norm].std()
min_cols = data.loc[data["isFraud"]==0 , cols_to_norm].min()
max_cols = data.loc[data["isFraud"]==0 , cols_to_norm].max()

data.loc[:, cols_to_norm] = (data[cols_to_norm] - min_cols) / (max_cols - min_cols)
```

    Index(['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5',
           'addr1', 'addr2', 'dist1', 'dist2',
           ...
           'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_24', 'id_25',
           'id_26', 'id_32'],
          dtype='object', length=401)



```python
print(min_cols)
print(max_cols)
data.head()
```

    TransactionDT     86400.000
    TransactionAmt        0.251
    card1              1000.000
    card2               100.000
    card3               100.000
                        ...    
    id_22                10.000
    id_24                11.000
    id_25               100.000
    id_26               100.000
    id_32                 0.000
    Length: 401, dtype: float64
    TransactionDT     1.581113e+07
    TransactionAmt    3.193739e+04
    card1             1.839600e+04
    card2             6.000000e+02
    card3             2.310000e+02
                          ...     
    id_22             4.400000e+01
    id_24             2.600000e+01
    id_25             5.480000e+02
    id_26             2.160000e+02
    id_32             3.200000e+01
    Length: 401, dtype: float64





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
      <th>2987000</th>
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
      <th>2987001</th>
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
      <th>2987002</th>
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
      <th>2987003</th>
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
      <th>2987004</th>
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
      <td>0.511628</td>
      <td>NaN</td>
      <td>0.774081</td>
      <td>0.078431</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2834 columns</p>
</div>



I saved the preprocessed data into a numpy file format and load it using the pytorch data loader.


```python
np.savez_compressed("ieee_fraud",ieee=data.as_matrix())
```

    /usr/local/var/pyenv/versions/anaconda3-2019.10/envs/ana3env/lib/python3.7/site-packages/ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
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
    'data_path' : 'ieee_fraud.npz',

    'log_path' : './dagmm/ieee_logs',
    'model_save_path' : './dagmm/ieee_models',
    'sample_path' : './dagmm/ieee_samples',
    'test_sample_path' : './dagmm/ieee_test_samples',
    'result_path' : './dagmm/ieee_results',

    'log_step' : 194//4,
    'sample_step' : 194,
    'model_save_step' : 194,
}
```


```python
solver = main(hyperparams(defaults))
accuracy, precision, recall, f_score = solver.test()
```

    590540
    2833


      0%|          | 0/1 [00:00<?, ?it/s]

    DaGMM
    DaGMM(
      (encoder): Sequential(
        (0): Linear(in_features=118, out_features=60, bias=True)
        (1): Tanh()
        (2): Linear(in_features=60, out_features=30, bias=True)
        (3): Tanh()
        (4): Linear(in_features=30, out_features=10, bias=True)
        (5): Tanh()
        (6): Linear(in_features=10, out_features=1, bias=True)
      )
      (decoder): Sequential(
        (0): Linear(in_features=1, out_features=10, bias=True)
        (1): Tanh()
        (2): Linear(in_features=10, out_features=30, bias=True)
        (3): Tanh()
        (4): Linear(in_features=30, out_features=60, bias=True)
        (5): Tanh()
        (6): Linear(in_features=60, out_features=118, bias=True)
      )
      (estimation): Sequential(
        (0): Linear(in_features=3, out_features=10, bias=True)
        (1): Tanh()
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=10, out_features=4, bias=True)
        (4): Softmax(dim=1)
      )
    )
    The number of parameters: 18783


    



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-16-db7e643ba5ba> in <module>
    ----> 1 solver = main(hyperparams(defaults))
          2 accuracy, precision, recall, f_score = solver.test()


    ~/notebook/anaconda_env/dagmm/main.py in main(config)
         23 
         24     if config.mode == 'train':
    ---> 25         solver.train()
         26     elif config.mode == 'test':
         27         solver.test()


    ~/notebook/anaconda_env/dagmm/solver.py in train(self)
         95                 input_data = self.to_var(input_data)
         96 
    ---> 97                 total_loss,sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
         98                 # Logging
         99                 loss = {}


    ~/notebook/anaconda_env/dagmm/solver.py in dagmm_step(self, input_data)
        160     def dagmm_step(self, input_data):
        161         self.dagmm.train()
    --> 162         enc, dec, z, gamma = self.dagmm(input_data)
        163 
        164         total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)


    /usr/local/var/pyenv/versions/anaconda3-2019.10/envs/ana3env/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    ~/notebook/anaconda_env/dagmm/model.py in forward(self, x)
         68     def forward(self, x):
         69 
    ---> 70         enc = self.encoder(x)
         71 
         72         dec = self.decoder(enc)


    /usr/local/var/pyenv/versions/anaconda3-2019.10/envs/ana3env/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    /usr/local/var/pyenv/versions/anaconda3-2019.10/envs/ana3env/lib/python3.7/site-packages/torch/nn/modules/container.py in forward(self, input)
         90     def forward(self, input):
         91         for module in self._modules.values():
    ---> 92             input = module(input)
         93         return input
         94 


    /usr/local/var/pyenv/versions/anaconda3-2019.10/envs/ana3env/lib/python3.7/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        539             result = self._slow_forward(*input, **kwargs)
        540         else:
    --> 541             result = self.forward(*input, **kwargs)
        542         for hook in self._forward_hooks.values():
        543             hook_result = hook(self, input, result)


    /usr/local/var/pyenv/versions/anaconda3-2019.10/envs/ana3env/lib/python3.7/site-packages/torch/nn/modules/linear.py in forward(self, input)
         85 
         86     def forward(self, input):
    ---> 87         return F.linear(input, self.weight, self.bias)
         88 
         89     def extra_repr(self):


    /usr/local/var/pyenv/versions/anaconda3-2019.10/envs/ana3env/lib/python3.7/site-packages/torch/nn/functional.py in linear(input, weight, bias)
       1368     if input.dim() == 2 and bias is not None:
       1369         # fused op is marginally faster
    -> 1370         ret = torch.addmm(bias, input, weight.t())
       1371     else:
       1372         output = input.matmul(weight.t())


    RuntimeError: size mismatch, m1: [3 x 2833], m2: [118 x 60] at /Users/distiller/project/conda/conda-bld/pytorch_1573049287641/work/aten/src/TH/generic/THTensorMath.cpp:197


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
