"""Tests for the PitchDetector class."""

import numpy as np

from audio_to_midi.pitch.detector import PitchDetector


class TestPitchDetectorDefaults:
    """Tests for PitchDetector default configuration."""

    def test_default_algorithm_is_pyin(self):
        """pyin should be the default algorithm for best real-world performance."""
        detector = PitchDetector()
        assert detector.algorithm == "pyin"

    def test_default_confidence_threshold_is_practical(self):
        """Default confidence should be 0.5 for practical use with various audio sources."""
        detector = PitchDetector()
        assert detector.confidence_threshold == 0.5


class TestPitchDetectorPyin:
    """Tests for pyin algorithm integration."""

    def test_pyin_algorithm_is_valid_option(self):
        """PitchDetector should accept 'pyin' as a valid algorithm."""
        detector = PitchDetector()
        # Should not raise - pyin should be a valid algorithm choice
        detector.configure(algorithm="pyin")
        assert detector.algorithm == "pyin"

    def test_pyin_detects_440hz_sine_wave(self):
        """pyin should correctly detect a 440 Hz sine wave (A4)."""
        detector = PitchDetector()
        detector.configure(
            sample_rate=44100,
            min_freq=80.0,
            max_freq=2000.0,
            confidence_threshold=0.5,
            algorithm="pyin",
        )

        # Generate a 440 Hz sine wave (A4)
        duration = 0.1  # 100ms
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        frequency, confidence = detector.detect_pitch(audio_data)

        assert frequency is not None, "Should detect a frequency"
        assert 430 <= frequency <= 450, f"Expected ~440 Hz, got {frequency}"
        assert confidence > 0.5, f"Expected confidence > 0.5, got {confidence}"

    def test_pyin_detects_multiple_frequencies(self):
        """pyin should accurately detect various frequencies."""
        sample_rate = 44100
        duration = 0.1

        test_frequencies = [261.63, 440.0, 880.0]  # C4, A4, A5

        for expected_freq in test_frequencies:
            # Fresh detector for each frequency to avoid smoothing effects
            detector = PitchDetector()
            detector.configure(
                sample_rate=sample_rate,
                min_freq=80.0,
                max_freq=2000.0,
                confidence_threshold=0.3,
                algorithm="pyin",
            )

            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = np.sin(2 * np.pi * expected_freq * t).astype(np.float32)

            frequency, confidence = detector.detect_pitch(audio_data)

            tolerance = expected_freq * 0.05  # 5% tolerance
            assert frequency is not None, f"Should detect {expected_freq} Hz"
            assert (
                abs(frequency - expected_freq) < tolerance
            ), f"Expected ~{expected_freq} Hz, got {frequency}"
