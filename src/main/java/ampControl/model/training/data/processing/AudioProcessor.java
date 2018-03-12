package ampControl.model.training.data.processing;

import ampControl.audio.processing.ProcessingResult;

/**
 * Interface for delivering a {@link ProcessingResult}.
 *
 * @author Christian Skärby
 */
public interface AudioProcessor {

	/**
	 * Get the result form the processing.
	 *
	 * @return a {@link ProcessingResult}
	 */
	ProcessingResult getResult();

}