package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link SoundnetFactory}
 *
 * @author Christian Skärby
 */
public class SoundnetFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer()
                .setInputSize(new int[] {1, 4410})
                .initialize(
                        ((tr, ev, is, np, md) -> new SoundnetFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
                );
    }
}