import unittest

import numpy as np
import pandas as pd

from airline_satisfaction.features import RATINGS
from airline_satisfaction.preprocessing import CATEGORICAL, DELAY
from airline_satisfaction.lgbm_exploration import AssociationPreprocessor, rating_keys


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.X=pd.DataFrame({'Age':np.arange(20,40),'Flight Distance':[100]*10+[200]*10,
            'Departure Delay in Minutes':[0]*20,DELAY:[0.]*20,
            **{c:[1]*10+[5]*10 for c in RATINGS},**{c:['a']*20 for c in CATEGORICAL}})
        self.y=np.array([0,1]*10)

    def test_route_profiles_and_counts_are_training_only(self):
        prep=AssociationPreprocessor(block='route_profile').fit(self.X,self.y)
        valid=self.X.iloc[:2].copy()
        valid['Inflight wifi service']=100
        valid.loc[1,'Flight Distance']=9999
        out=prep.transform(valid)
        expected=(10*1+20*3)/30
        self.assertAlmostEqual(out.loc[0,'RP_mean_Inflight wifi service'],expected,places=6)
        self.assertEqual(out.loc[1,'RP_mean_Inflight wifi service'],3.)
        self.assertEqual(out.loc[1,'RP_support_TE_distance'],0.)
        self.assertEqual(out.loc[0,'RP_support_TE_distance'],10.)

    def test_rating_keys_and_crossfit_are_finite_and_reproducible(self):
        self.assertEqual(rating_keys(self.X).shape[1],39)
        a=AssociationPreprocessor(block='rating_context').fit_transform(self.X,self.y)
        b=AssociationPreprocessor(block='rating_context').fit_transform(self.X,self.y)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(np.isfinite(a.filter(regex='^RT_').to_numpy()).all())


if __name__=='__main__':
    unittest.main()
