package ampcontrol.model.training.data.processing;

import java.nio.file.Path;
import java.util.List;
import java.util.function.Supplier;

/**
 * {@link Supplier} which returns {@link Path Paths} in a sequence according to a given {@link List}. The same
 * {@link Path} is repeated a number of times as decided by nrToHold.
 * <br><br>
 * Intended use is to e.g. provide 10x100 ms windows of a 1s long wav-file in sequence when combined with a
 * {@link WindowedConsecutiveSamplingInfo}.
 *
 * @author Christian Skärby
 */
public class SequentialHoldFileSupplier implements Supplier<Path> {
	
	private final List<Path> files;
	private final int nrToHold;

	private int fileNr = 0;
	private int holdNr = 0;

	/**
	 * Constructor
	 * @param files files to supply
	 * @param nrToHold how many times to repeat each file
	 */
	public SequentialHoldFileSupplier(List<Path> files, int nrToHold) {
		this(files,nrToHold,0);
	}

	/**
	 * Constructor
	 * @param files files to supply
	 * @param nrToHold how many times to repeat each file
	 * @param startInd starting index in files
	 */
	public SequentialHoldFileSupplier(List<Path> files, int nrToHold, int startInd) {
		this.files = files;
		this.nrToHold = nrToHold;
		final int positiveStartInd = startInd < 0 ? -startInd : startInd;
		fileNr = positiveStartInd % files.size();
		if(files.isEmpty()) {
			throw new IllegalArgumentException("No files given!");
		}
		if(nrToHold < 1) {
			throw new IllegalArgumentException("nrToHold must be > 0!");
		}
	}

	@Override
	public synchronized Path get() {
		if(fileNr == files.size()) {
			fileNr = 0;
			holdNr = 0;
		}
		Path toRet = files.get(fileNr);
		holdNr++;
		if(holdNr == nrToHold) {
			holdNr = 0;
			fileNr++;
		}
		
		return toRet;
	}

}
