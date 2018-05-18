package ampcontrol.model.training.data.processing;

import ampcontrol.audio.processing.*;

import java.util.Arrays;
import java.util.function.Supplier;

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
	 * @param resultSupplier Supplier of {@link ProcessingResult.Factory}.
	 */
	public SilenceProcessor(int windowSize, final Supplier<ProcessingResult.Factory> resultSupplier) {
		final double[] silenceSample = new double[windowSize];
		Arrays.fill(silenceSample, 0);
		ProcessingResult.Factory factory = resultSupplier.get();
		silence = factory.create(new SingletonDoubleInput(silenceSample));
	}

	@Override
	public ProcessingResult getResult() {
		return silence;
	}
	
	public static void main(String[] args) {
		double[][] data = new SilenceProcessor(44100 * 1000 / (1000/200) / 1000,
				() -> new Pipe(
						new Spectrogram(512, 32),
						new UnitStdZeroMean())).getResult().stream().findFirst()
				.orElseThrow(() -> new RuntimeException("No output!"));

		System.out.println("shape: " + data.length + ", " + data[0].length);
		System.out.println("data: " + Arrays.deepToString(data));
	}

}
