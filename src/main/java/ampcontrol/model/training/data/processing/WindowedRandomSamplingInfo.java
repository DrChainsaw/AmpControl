package ampcontrol.model.training.data.processing;

import java.nio.file.Path;
import java.util.Random;
import java.util.function.Function;

/**
 * Creates an {@link AudioSamplingInfo} with a given length but where start point is random.
 *
 * @author Christian Skärby
 */
public class WindowedRandomSamplingInfo implements Function<Path, AudioSamplingInfo> {

	private final double clipLength;
	private final double windowSize;
	private final Random rng;

	/**
	 * Constructor
	 * @param clipLengthMs (Assumed) length of clips in milliseconds
	 * @param windowSizeMs Wanted window size in milliseconds
	 * @param rng Random number generator
	 */
	public WindowedRandomSamplingInfo(int clipLengthMs, int windowSizeMs, Random rng) {
		this.clipLength = clipLengthMs / 1000d;
		this.windowSize = windowSizeMs / 1000d;
		this.rng = rng;
	}

	@Override
	public AudioSamplingInfo apply(Path file) {
		final double start = rng.nextDouble() * (clipLength - windowSize);
		return new AudioSamplingInfo(start, windowSize);
	}
	
	

}
