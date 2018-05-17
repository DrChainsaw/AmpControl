package ampcontrol.model.training.data;

import java.nio.file.Path;

/**
 * Interface for creating a {@link DataProvider}.
 *
 * @author Christian Skärby
 */
public interface DataProviderBuilder {

	/**
	 * Build a new {@link DataProvider} instance
	 * @return a new {@link DataProvider} instance
	 */
	DataProvider createProvider();

	/**
	 * Add a label for which features shall be created
	 * @param label a label
	 */
	void addLabel(String label);

	/**
	 * Add a file with data
	 * @param file
	 * @return the builder instance
	 */
	DataProviderBuilder addFile(Path file);

	/**
	 * Returns the number of files given to this builder
	 * @return the number of files given to this builder
	 */
	int getNrofFiles();

}
