import pytest
import pandas as pd
import numpy as np
from anomaly_agent.detection import Detector
from anomaly_agent.config import Config

class MockConfig:
    def __init__(self, required=2, window=3):
        self.detection = type('obj', (object,), {
            'persistence': {'required': required, 'window': window}
        })

@pytest.fixture
def detector():
    cfg = MockConfig()
    return Detector(cfg)

def test_persistence_blip_suppressed(detector):
    # 1-day blip in 5 days
    flags = pd.Series([False, False, True, False, False], 
                      index=pd.date_range('2023-01-01', periods=5))
    persistent = detector._apply_persistence(flags)
    assert not persistent.any(), "1-day blip should be suppressed"

def test_persistence_shift_alerts(detector):
    # 3-day shift in 5 days
    # Day 3, 4, 5 are flagged
    flags = pd.Series([False, False, True, True, True], 
                      index=pd.date_range('2023-01-01', periods=5))
    persistent = detector._apply_persistence(flags)
    
    # With N=2, M=3:
    # Day 3: sum=1 (False)
    # Day 4: sum=2 (True)
    # Day 5: sum=3 (True)
    assert not persistent.iloc[2], "First day of shift might be suppressed if window starts there"
    assert persistent.iloc[3], "Second day of shift should alert"
    assert persistent.iloc[4], "Third day of shift should alert"

def test_persistence_intermittent_alerts(detector):
    # N=2, M=3. Flag, No-Flag, Flag
    flags = pd.Series([True, False, True], 
                      index=pd.date_range('2023-01-01', periods=3))
    persistent = detector._apply_persistence(flags)
    
    # Day 1: sum=1
    # Day 2: sum=1
    # Day 3: sum=2 (True)
    assert persistent.iloc[2], "Intermittent flag within window should alert on the second flag"
