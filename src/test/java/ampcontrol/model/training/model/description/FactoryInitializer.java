package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.MockDataSetIterator;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Test class for avoid some overhead when creating the factories which reside in this package.
 *
 * @author Christian Skärby
 */
class FactoryInitializer {

    // Create setters if needed...
    private CachingDataSetIterator train = new CachingDataSetIterator(new MockDataSetIterator() {
        @Override
        public int totalOutcomes() {
            return 4;
        }
    });
    private CachingDataSetIterator eval = new CachingDataSetIterator(new MockDataSetIterator());
    private int[] inputSize = {128,128,2};
    private String namePrefix = "testDummy_";
    private Path modelSaveDir = Paths.get(".");

    interface FactoryFactory {
        void init(CachingDataSetIterator tr, CachingDataSetIterator ev, int[] is, String np, Path md);
    }

    void initialize(FactoryFactory ff) {
        ff.init(train,eval,inputSize, namePrefix, modelSaveDir);
    }

    FactoryInitializer setInputSize(int[] inputSize) {
        this.inputSize = inputSize;
        return this;
    }
}
