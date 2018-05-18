package ampcontrol.audio;

import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.SingletonDoubleInput;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

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
        final Supplier<ProcessingResult.Factory> processingSupplier = () -> new SplitInputProcessing(2, nrofChannels);
        final ProcessingResult.Factory proc = processingSupplier.get();


        final ClassifierInputProvider.Updatable inputProvider = new Cnn2DInputProvider(mockBuffer, processingSupplier);

        inputProvider.updateInput();
        final ProcessingResult res0 = proc.create(new SingletonDoubleInput(audioFrames[0]));
        final INDArray expected0 = Nd4j.create(res0.stream().findFirst().get());
        for(int channel = 0; channel < nrofChannels; channel++) {
            assertEquals("Incorrect model input!", expected0, inputProvider.getModelInput().get(NDArrayIndex.point(0), NDArrayIndex.point(channel)));
        }

        mockBuffer.advance();
        inputProvider.updateInput();
        final ProcessingResult res1 = proc.create(new SingletonDoubleInput(audioFrames[1]));
        final INDArray expected1 = Nd4j.create(res1.stream().findFirst().get());
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

    private static class SplitInputProcessing implements ProcessingResult.Factory {

        private final int splitSize;
        private final int nrofOutputDupes;

        public SplitInputProcessing(int splitSize, int nrofOutputDupes) {
            this.splitSize = splitSize;
            this.nrofOutputDupes = nrofOutputDupes;
        }

        @Override
        public ProcessingResult create(ProcessingResult input) {
            return new Result(input);
        }

        private final class Result implements ProcessingResult {

            private final ProcessingResult input;

            public Result(ProcessingResult input) {
                this.input = input;
            }

            @Override
            public Stream<double[][]> stream() {
                return Collections.nCopies(nrofOutputDupes, input.stream().map(inputArr -> {
                    final int nrofSamplesPerFrame = inputArr[0].length / splitSize;
                    final double[][] result = new double[splitSize][nrofSamplesPerFrame];
                    for (int i = 0; i < splitSize; i++) {
                        for (int j = 0; j < nrofSamplesPerFrame; j++) {
                            result[i][j] = inputArr[0][j + i * splitSize];
                        }
                    }
                    return result;
                }).findFirst().orElseThrow(() -> new RuntimeException("No inputs!"))).stream();
            }
        }

        @Override
        public String name() {
            return null;
        }
    }
}