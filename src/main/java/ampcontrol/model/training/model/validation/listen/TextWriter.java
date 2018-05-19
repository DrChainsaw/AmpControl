package ampcontrol.model.training.model.validation.listen;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Facade interface to be able to stub out file writing in tests
 *
 * @author Christian Skärby
 */
public interface TextWriter {

    /**
     * Factory interface
     */
    interface Factory {
        /**
         * Create a TextWriter to write to the given file
         * @param path path to file
         * @return a new TextWriter
         * @throws IOException
         */
        TextWriter create(Path path) throws IOException;
    }

    /**
     * Write the given string as a new line in a file
     *
     * @param str string to write
     */
    void write(String str) throws IOException;

    /**
     * Closes the file for writing
     */
    default void close() throws IOException {
        //Ignore
    }
}
