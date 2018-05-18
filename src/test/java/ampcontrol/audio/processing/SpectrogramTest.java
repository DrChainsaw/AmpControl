package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Test cases for Spectrogram
 *
 * @author Christian Skärby
 */
public class SpectrogramTest {

    /**
     * Test that the output has the properties of a spectrogram
     */
    @Test
    public void receive() {
        final int nrofFrames = 7;
        final int fftWindowSize = 256;
        final int perFrameFreqShift = 7;
        final List<Integer> freqs = Arrays.asList(11, 23);
        // Create a signal with different frequency components in each time window
        double[] cosSum = IntStream.range(0, nrofFrames * fftWindowSize)
                .mapToDouble(i -> freqs.stream()
                        .map(freq -> freq + perFrameFreqShift * Math.floor(i / fftWindowSize))
                        .map(freq -> freq * 2 * Math.PI / fftWindowSize)
                        .mapToDouble(freq -> Math.cos(freq * i)).sum())
                .toArray();


        final ProcessingResult.Factory specgram = new Spectrogram(fftWindowSize, fftWindowSize);
        final ProcessingResult res = specgram.create(new SingletonDoubleInput(new double[][]{cosSum}));

        final double[][] specgramData = res.stream().findFirst().get();
        assertEquals("Incorrect number of frames!", nrofFrames, specgramData.length);

        for(int frameNr = 0; frameNr < nrofFrames; frameNr++) {
            for (int i = 0; i < freqs.size(); i++) {
                final int argmax = argMax(specgramData[frameNr]);
                specgramData[frameNr][argmax] = 0;
                final Integer expectedFreqListMember = argmax - frameNr * perFrameFreqShift;
                assertTrue("Frequency component " + expectedFreqListMember + " not present in signal!", freqs.contains(expectedFreqListMember));
            }
        }
    }

    /**
     * Test that the expected number of frames is generated when overlapping is used
     */
    @Test
    public void overlapping() {
        final int expectedNrofFrames = 11;
        final int fftSize = 32;
        final int stride = 8;
        final int signalSize = (expectedNrofFrames - 1) * stride + fftSize;
        double[] signal = IntStream.range(0, signalSize).mapToDouble(i -> i).toArray();

        final ProcessingResult.Factory specgram = new Spectrogram(fftSize, stride);
        final ProcessingResult res = specgram.create(new SingletonDoubleInput(new double[][]{signal}));

        final double[][] specgramData = res.stream().findFirst().get();
        assertEquals("Incorrect number of frames!", expectedNrofFrames, specgramData.length);
    }

    /**
     * Test that name can be used to create an identical instance
     */
    @Test
    public void name() {
        final String expectedName = new Spectrogram(256, 128).name();
        assertEquals("Inconsistent naming!",expectedName, new Spectrogram(expectedName).name());
    }

    /**
     * Test that name and nameStatic is consistent
     */
    @Test
    public void nameStatic() {
        assertTrue("Inconsistent naming!", new Spectrogram(32, 2).name().contains(Spectrogram.nameStatic()));
    }


    private static int argMax(double[] in) {
        int maxIdx = 0;

        for (int i = 1; i < in.length; ++i) {
            if (in[i] > in[maxIdx]) {
                maxIdx = i;
            }
        }

        return maxIdx;
    }
}