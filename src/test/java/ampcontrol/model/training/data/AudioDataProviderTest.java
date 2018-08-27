package ampcontrol.model.training.data;

import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.model.training.data.processing.AudioProcessor;
import org.junit.Test;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AudioDataProvider}.
 *
 * @author Christian Skärby
 */
public class AudioDataProviderTest {

    /**
     * Test that the involved objects are correctly initialized and that the expected result is produced.
     */
    @Test
    public void generateData() {
        final String label1 = "label1";
        final String label2 = "label2";
        final ProcessingResult res1 = () -> null;
        final ProcessingResult res2 = () -> null;
        final MockAudioProcessorBuilder builder1 = new MockAudioProcessorBuilder(() -> res1);
        final MockAudioProcessorBuilder builder2 = new MockAudioProcessorBuilder(() -> res2);
        final Map<String, AudioDataProvider.AudioProcessorBuilder> labelToBuilder = new HashMap<>();
        labelToBuilder.put(label1, builder1);
        labelToBuilder.put(label2, builder2);
        final List<Path> paths1 = Stream.of("gerf", "gtth", "jyujuy").map(str -> Paths.get(label1 + File.separator + str)).collect(Collectors.toList());
        final List<Path> paths2 = Stream.of("ewrt", "vcdfbj").map(str -> Paths.get(label2 + File.separator + str)).collect(Collectors.toList());
        final List<Path> allpaths = Stream.concat(paths1.stream(), paths2.stream()).collect(Collectors.toList());

        final DataProvider pr = new AudioDataProvider(allpaths, labelToBuilder, new ToggleSupplier(label1,label2));
        builder1.assertExpectedPaths(paths1);
        builder2.assertExpectedPaths(paths2);

        DataProvider.TrainingData data = pr.generateData().limit(1).findAny().get();
        assertEquals("Incorrect label!", label1, data.getLabel());
        assertEquals("Incorrect result!", res1, data.result());

        data = pr.generateData().limit(1).findAny().get();
        assertEquals("Incorrect label!", label2, data.getLabel());
        assertEquals("Incorrect result!", res2, data.result());

        data = pr.generateData().limit(1).findAny().get();
        assertEquals("Incorrect label!", label1, data.getLabel());
        assertEquals("Incorrect result!", res1, data.result());

        data = pr.generateData().limit(1).findAny().get();
        assertEquals("Incorrect label!", label2, data.getLabel());
        assertEquals("Incorrect result!", res2, data.result());

    }


    private static class MockAudioProcessorBuilder implements AudioDataProvider.AudioProcessorBuilder {

        private final List<Path> paths = new ArrayList<>();
        private final AudioProcessor proc;

        public MockAudioProcessorBuilder(AudioProcessor proc) {
            this.proc = proc;
        }

        @Override
        public AudioProcessor build() {
            return proc;
        }

        @Override
        public AudioDataProvider.AudioProcessorBuilder add(Path file) {
            paths.add(file);
            return this;
        }

        private void assertExpectedPaths(List<Path> expected) {
            assertEquals("Incorrect paths!", expected, paths);
        }
    }

    private static class ToggleSupplier implements Supplier<String> {
        private final String str1;
        private final String str2;
        private boolean toggle = false;

        public ToggleSupplier(String str1, String str2) {
            this.str1 = str1;
            this.str2 = str2;
        }

        @Override
        public String get() {
            toggle = !toggle;
            return toggle ? str1: str2;
        }
    }
}