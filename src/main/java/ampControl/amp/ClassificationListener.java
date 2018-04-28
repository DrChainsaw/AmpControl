package ampControl.amp;

import ampControl.amp.midi.MidiProgChangeFactory;
import ampControl.amp.midi.PodXtFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Interface to receive a classification in the form of a vector where each element corresponds to the estimated
 * probability of the associated label (e.g. Rythm, Lead, Cat, etc.).
 *
 * @author Christian Skärby
 */
public interface ClassificationListener {

    /**
     * Factory interface
     */
    interface Factory {
        /**
         * Create a {@link ClassificationListener}.
         *
         * @return
         */
        ClassificationListener create();
    }

    /**
     * Indicate label probabilities
     *
     * @param probabilities
     */
    void indicateAudioClassification(INDArray probabilities) ;

    /**
     * Creates a map of JCommander commands to ClassificationListener factories
     *
     * @return
     */
    static Map<String, Factory> getFactoryCommands() {
        Map<String, Factory> factoryMap = new LinkedHashMap<>();
        factoryMap.put("-podXt", new PodXtFactory());
        factoryMap.put("-midiPrgChange", new MidiProgChangeFactory());
        factoryMap.put("-dummy", () -> new DummyClassifictionListener());
        factoryMap.put("-print", new PrintClassificationListener.Factory());

        return factoryMap;
    }
}
