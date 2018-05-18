package ampcontrol.audio.asio;

import org.junit.Test;

import java.util.Collections;
import java.util.stream.DoubleStream;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link AsioAudioInputBuffer}.
 *
 * @author Christian Skärby
 */
public class AsioAudioInputBufferTest {

    /**
     * Test that the circular buffer holds the last desiredNrofSamples it has gotten from the input channel
     */
    @Test
    public void getAudio() {
        final MockInputChannel inputChannel = new MockInputChannel();
        final int bufferSize = 10;
        final int desiredNrofSamples = 15;
        final AsioAudioInputBuffer buffer = new AsioAudioInputBuffer(inputChannel,bufferSize, desiredNrofSamples);


        inputChannel.setNewBuffer(createBuffer(bufferSize, 0));
        buffer.bufferSwitch(0,0, Collections.emptySet());

        inputChannel.setNewBuffer(createBuffer(bufferSize, bufferSize));
        buffer.bufferSwitch(0,0, Collections.emptySet());

        final double[] expectedSamples = new double[desiredNrofSamples];
        for(int i = 0; i < desiredNrofSamples; i++) {
            float f = (i + desiredNrofSamples - bufferSize) / (float)bufferSize;
            expectedSamples[i] = f;
        }
        double[] rescaledOutput = DoubleStream.of(buffer.getAudio()).map(d -> d / Short.MAX_VALUE).toArray();
        assertArrayEquals("Incorrect output!", expectedSamples, rescaledOutput, 1e-10);
    }

    private static float[] createBuffer(int size, int offs) {
        float[] output = new float[size];
        for(int i = 0; i < size; i++) {
            output[i] = (i + offs) / (float)size;
        }
        return output;
    }
}