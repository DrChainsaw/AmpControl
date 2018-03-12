package ampControl.model.training.data.processing;

import java.util.Arrays;
import java.util.function.Supplier;

import ampControl.audio.processing.Pipe;
import ampControl.audio.processing.ProcessingResult;
import ampControl.audio.processing.Spectrogram;
import ampControl.audio.processing.UnitStdZeroMean;

/**
 * {@link AudioProcessor} which returns silence (all zeroes)
 *
 * @author Christian Skärby
 */
public class SilenceProcessor implements AudioProcessor {
	
	private final ProcessingResult silence;

	/**
	 * Constructor
	 * @param windowSize Size of an audio sample in number if samples
	 * @param resultSupplier Supplier of {@link ProcessingResult.Processing}.
	 */
	public SilenceProcessor(int windowSize, final Supplier<ProcessingResult.Processing> resultSupplier) {
		final double[] silenceSample = new double[windowSize];
		Arrays.fill(silenceSample, 0);
		ProcessingResult.Processing res = resultSupplier.get();
		res.receive(new double[][] {silenceSample});
		silence = res;
	}

	@Override
	public ProcessingResult getResult() {
		return silence;
	}
	
	public static void main(String[] args) {
		double[][] data = new SilenceProcessor(44100 * 1000 / (1000/200) / 1000,
				() -> new Pipe(
						new Spectrogram(512, 32),
						new UnitStdZeroMean())).getResult().get().get(0);

		System.out.println("shape: " + data.length + ", " + data[0].length);
		System.out.println("data: " + Arrays.deepToString(data));
	}

}
