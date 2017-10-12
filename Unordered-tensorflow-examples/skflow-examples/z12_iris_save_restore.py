#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

from sklearn import datasets, metrics, cross_validation
from tensorflow.contrib import skflow

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
    test_size=0.2, random_state=42)

classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))

# Clean checkpoint folder if exists
try:
    shutil.rmtree('/home/ubuntu/workspace/skflow_examples/iris/iris_custom_model')
except OSError:
    pass

# Save model, parameters and learned variables.
classifier.save('/home/ubuntu/workspace/skflow-examples/iris_custom_model')
classifier = None

## Restore everything
new_classifier = skflow.TensorFlowEstimator.restore('/home/ubuntu/workspace/skflow-examples/iris_custom_model')
score = metrics.accuracy_score(y_test, new_classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))
