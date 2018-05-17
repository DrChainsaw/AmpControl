package ampcontrol.audio;

import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.SingletonDoubleInput;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * {@link ClassifierInputProvider} for 2D CNN. Takes samples from an audio buffer and runs them through supplied
 * {@link ProcessingResult.Factory} and puts the output in an {@link INDArray}.
 *
 * @author Christian Skärby
 */
public class Cnn2DInputProvider implements ClassifierInputProvider.Updatable {

    private final AudioInputBuffer audioBuffer;
    private final INDArray output;
    private final Supplier<ProcessingResult.Factory> resultSupplier;

    public Cnn2DInputProvider(
            AudioInputBuffer audioBuffer,
            Supplier<ProcessingResult.Factory> resultSupplier) {
        this.audioBuffer = audioBuffer;
        this.resultSupplier = resultSupplier;

        List<double[][]> result = getPostProcessedInput();
        this.output = Nd4j.create(new int[] {1, result.size(), result.get(0).length, result.get(0)[0].length}, 'f');
    }

    @Override
    public INDArray getModelInput() {
        return output.dup();
    }

    @Override
    public void updateInput() {
        List<double[][]> nextInput = getPostProcessedInput();
        for(int inputInd = 0; inputInd < nextInput.size(); inputInd++) {
            double[][] oneInput = nextInput.get(inputInd);
            for (int timeInd = 0; timeInd < oneInput.length; timeInd++) {
                for (int freqInd = 0; freqInd < oneInput[timeInd].length; freqInd++) {
                    output.putScalar(0, inputInd, timeInd, freqInd, oneInput[timeInd][freqInd]);
                }
            }
        }
    }

    private List<double[][]> getPostProcessedInput() {

        ProcessingResult.Factory next = resultSupplier.get();
        double[] audioFrame = audioBuffer.getAudio();
        ProcessingResult res = next.create(new SingletonDoubleInput(audioFrame));
        return res.stream().collect(Collectors.toList());

    }
}
