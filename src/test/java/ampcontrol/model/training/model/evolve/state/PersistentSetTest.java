package ampcontrol.model.training.model.evolve.state;

import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link PersistentSet}
 */
public class PersistentSetTest {

    /**
     * Test persisting a set and then loading it
     */
    @Test
    public void persist() {
        final Path baseDir = Paths.get("src", "test", "resources", "PersistentSetTest");
        final String fileName = Paths.get(baseDir.toString(), "get.json").toString();

        try {
            final PersistentSet<String> expected = new PersistentSet<>(fileName, Stream.of("a", "test", "set").collect(Collectors.toSet()));
            assertFalse("Set shall not be empty!", expected.get().isEmpty());
            expected.save();
            final PersistentSet<String> actual = new PersistentSet<>(fileName, Collections.emptySet());
            assertEquals("Set did not persist!", expected.get(), actual.get());
            new File(fileName).delete();
            Files.delete(baseDir);
        } catch (IOException e) {
            throw new IllegalStateException();
        }

    }

}