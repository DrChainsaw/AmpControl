package ampcontrol.model.training.data.processing;

import ampcontrol.model.training.data.state.StateFactory;
import org.apache.commons.lang.mutable.MutableInt;

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

	private final Supplier<MutableInt> fileNr;
	private final Supplier<MutableInt> holdNr;

	/**
	 * Constructor
	 * @param files files to supply
	 * @param nrToHold how many times to repeat each file
	 */
	public SequentialHoldFileSupplier(List<Path> files, int nrToHold, StateFactory stateFactory) {
		this(files,nrToHold,0, stateFactory);
	}

	/**
	 * Constructor
	 * @param files files to supply
	 * @param nrToHold how many times to repeat each file
	 * @param startInd starting index in files
	 */
	public SequentialHoldFileSupplier(List<Path> files, int nrToHold, int startInd, StateFactory stateFactory) {
		this.files = files;
		this.nrToHold = nrToHold;
		final int positiveStartInd = startInd < 0 ? -startInd : startInd;

		fileNr = stateFactory.createNewStateReference(mutInt -> new MutableInt(mutInt.intValue()), new MutableInt(positiveStartInd % files.size()));
		holdNr = stateFactory.createNewStateReference(mutInt -> new MutableInt(mutInt.intValue()), new MutableInt(0));
		if(files.isEmpty()) {
			throw new IllegalArgumentException("No files given!");
		}
		if(nrToHold < 1) {
			throw new IllegalArgumentException("nrToHold must be > 0!");
		}
	}

	@Override
	public synchronized Path get() {
		if(fileNr() == files.size()) {
			fileNr.get().setValue(0);
			holdNr.get().setValue(0);
		}
		Path toRet = files.get(fileNr());
		holdNr.get().increment();
		if(holdNr() == nrToHold) {
			holdNr.get().setValue(0);
			fileNr.get().increment();
		}
		
		return toRet;
	}

	private int fileNr() {
		return fileNr.get().intValue();
	}

	private int holdNr() {
		return holdNr.get().intValue();
	}
}
