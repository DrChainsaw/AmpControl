package ampcontrol.model.training.data.processing;

import org.junit.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link SequentialHoldFileSupplier}
 *
 * @author Christian Skärby
 */
public class SequentialHoldFileSupplierTest {

    /**
     * Test that files are supplied in the expected sequence.
     */
    @Test
    public void get() {
        final int startInd = 1;
        final int nrToHold = 3;
        final List<Path> fakeFiles = Arrays.asList(
                Paths.get("fefew"),
                Paths.get("tregg")
        );
        final List<Path> expected = Stream.concat(
                Collections.nCopies(nrToHold, fakeFiles.get(startInd)).stream(),
                Collections.nCopies(nrToHold, fakeFiles.get(startInd - 1)).stream()
        ).collect(Collectors.toList());

        final Supplier<Path> fileSupplier = new SequentialHoldFileSupplier(fakeFiles, nrToHold, startInd);

        final List<Path> actual = Stream.generate(fileSupplier).limit(expected.size()).collect(Collectors.toList());
        assertEquals("Incorrect sequence!", expected,actual);
    }
}