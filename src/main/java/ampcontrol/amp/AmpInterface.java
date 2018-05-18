package ampcontrol.amp;

import ampcontrol.admin.service.Service;
import ampcontrol.admin.service.control.SubscriptionRegistry;
import ampcontrol.amp.midi.MidiProgChangeFactory;
import ampcontrol.amp.midi.PodXtFactory;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Interface towards an amplifier which shall be controlled by this application.
 *
 * @author Christian Skärby
 */
public interface AmpInterface extends ClassificationListener, Service {

    interface Factory {

        /**
         * Creates a new {@link AmpInterface} instance
         * @return a new {@link AmpInterface} instance
         */
        AmpInterface create();
    }


    @Override
    default void stop() {/* May be ignored*/}

    @Override
    default void registerTo(SubscriptionRegistry subscriptionRegistry) {/* May be ignored*/}

    /**
     * Creates a map of JCommander commands to ClassificationListener factories
     *
     * @return
     */
    static Map<String, AmpInterface.Factory> getFactoryCommands() {
        Map<String, AmpInterface.Factory> factoryMap = new LinkedHashMap<>();
        factoryMap.put("-podXt", new PodXtFactory());
        factoryMap.put("-midiPrgChange", new MidiProgChangeFactory());
        factoryMap.put("-dummy", () -> new DummyClassifictionListener());
        factoryMap.put("-print", new PrintClassificationListener.Factory());

        return factoryMap;
    }
}
