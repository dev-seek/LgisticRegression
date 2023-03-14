# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# %%
X,y,centre = make_blobs(n_samples=1000,n_features=2,centers=2,random_state=0,return_centers="True")
X,y,centre

# %%
plt.scatter(X[:,0],X[:,1],c=y)
plt.scatter(centre[:,0],centre[:,1])
plt.legend()

# %%
X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.33,random_state=42)

# %%
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)

# %%
model.intercept_ ,model.coef_[0]
arr = zip(model.coef_ , model.intercept_)
array = list(arr)
array

# %%
X_gen = np.linspace(-6,7)
plt.scatter(X[:,0],X[:,1],c=y)
plt.scatter(centre[:,0],centre[:,1])
for i ,(coef,intercept) in enumerate(zip(model.coef_ , model.intercept_)):
    x2 = -(coef[0]/coef[1])*X_gen - (intercept/coef[1])
    plt.plot(X_gen,x2)
# plt.legend()

