package ampControl.audio;

import ampControl.audio.processing.ProcessingResult;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.*;

/**
 * Test cases for {@link Cnn2DInputProvider}.
 *
 * @author Christian Skärby
 */
public class Cnn2DInputProviderTest {

    /**
     * Test that input is created correctly
     */
    @Test
    public void getModelInput() {
        final int nrofChannels = 3;
        final double[][] audioFrames = new double[][] {{1,2,3,4},{5,6,7,8}};
        final MockAudioInputBuffer mockBuffer = new MockAudioInputBuffer(audioFrames);
        final Supplier<ProcessingResult.Processing> processingSupplier = () -> new SplitInputProcessing(2, nrofChannels);
        final ProcessingResult.Processing proc = processingSupplier.get();


        final ClassifierInputProvider.Updatable inputProvider = new Cnn2DInputProvider(mockBuffer, processingSupplier);

        inputProvider.updateInput();
        proc.receive(new double[][] {audioFrames[0]});
        final INDArray expected0 = Nd4j.create(proc.get().get(0));
        for(int channel = 0; channel < nrofChannels; channel++) {
            assertEquals("Incorrect model input!", expected0, inputProvider.getModelInput().get(NDArrayIndex.point(0), NDArrayIndex.point(channel)));
        }

        mockBuffer.advance();
        inputProvider.updateInput();
        proc.receive(new double[][] {audioFrames[1]});
        final INDArray expected1 = Nd4j.create(proc.get().get(0));
        for(int channel = 0; channel < nrofChannels; channel++) {
            assertEquals("Incorrect model input!", expected1, inputProvider.getModelInput().get(NDArrayIndex.point(0), NDArrayIndex.point(channel)));
        }
    }

    private static class MockAudioInputBuffer implements AudioInputBuffer {
        private final double[][] audioFrames;
        private int cnt;

        public MockAudioInputBuffer(double[][] audioFrames) {
            this.audioFrames = audioFrames;
            cnt = 0;
        }

        @Override
        public double[] getAudio() {
            return audioFrames[cnt];
        }

        private void advance() {
            cnt++;
        }
    }

    private static class SplitInputProcessing implements ProcessingResult.Processing {

        private final int splitSize;
        private final int nrofOutputDupes;
        private double[][] result;

        public SplitInputProcessing(int splitSize, int nrofOutputDupes) {
            this.splitSize = splitSize;
            this.nrofOutputDupes = nrofOutputDupes;
        }

        @Override
        public void receive(double[][] input) {
            final int nrofSamplesPerFrame = input[0].length / splitSize;
            result = new double[splitSize][nrofSamplesPerFrame];
            for(int i = 0; i< splitSize; i++) {
                for(int j = 0 ; j < nrofSamplesPerFrame; j++) {
                    result[i][j] = input[0][j + i*splitSize];
                }
            }
        }

        @Override
        public String name() {
            return null;
        }

        @Override
        public List<double[][]> get() {
            return Collections.nCopies(nrofOutputDupes, result);
        }
    }
}