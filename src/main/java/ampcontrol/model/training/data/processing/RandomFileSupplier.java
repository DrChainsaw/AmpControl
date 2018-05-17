package ampcontrol.model.training.data.processing;

import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * {@link Supplier} which provides a random {@link Path} from a given list.
 *
 * @author Christian Skärby
 */
class RandomFileSupplier implements Supplier<Path> {
	private final Random rng;
	private final List<Path> files;
	
	RandomFileSupplier(Random rng, List<Path> files) {
		this.rng = rng;
		this.files = files;
	}

	@Override
	public Path get() {
		return files.get(rng.nextInt(files.size()));
	}
	
	

}
