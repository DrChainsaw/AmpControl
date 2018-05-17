package ampcontrol.model.training.model.validation.listen;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;

/**
 * Wraps a {@link BufferedWriter}
 *
 * @author Christian Skärby
 */
public class BufferedTextWriter implements TextWriter {

    private final BufferedWriter writer;

    public final static Factory defaultFactory = path -> new BufferedTextWriter(Files.newBufferedWriter(path));

    public BufferedTextWriter(BufferedWriter writer) {
        this.writer = writer;
    }

    @Override
    public void write(String str) throws IOException {
        writer.write(str);
    }

    @Override
    public void close() throws IOException {
        writer.close();
    }
}
