package ampcontrol.admin.service.control.mqtt;

import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MqttCallbackMap}
 *
 */
public class MqttCallbackMapTest {

    /**
     * Test that events for which no action is mapped does not cause any problem
     *
     */
    @Test
    public void noActionMapped() {
        final MqttCallbackMap map = new MqttCallbackMap(str -> {});
        final String msg1Str = "msg1";
        final MqttMessage msg1 = new MqttMessage(msg1Str.getBytes());
        map.messageArrived("", msg1);
        map.connectionLost(new Throwable());
    }

    /**
     * Test message to action mapping
     */
    @Test
    public void mapAction() {
        final MqttCallbackMap map = new MqttCallbackMap(topic -> {});
        final String msg1Str = "msg1";
        final String msg2Str = "msg2";
        final MqttMessage msg1 = new MqttMessage(msg1Str.getBytes());
        final MqttMessage msg2 = new MqttMessage(msg2Str.getBytes());
        final ActionProbe probe1 = new ActionProbe();
        final ActionProbe probe2 = new ActionProbe();
        map.registerSubscription(msg1Str, probe1);
        map.registerSubscription(msg2Str, probe2);

        probe1.assertCalled(0);
        probe2.assertCalled(0);

        map.messageArrived("", msg1);

        probe1.assertCalled(1);
        probe2.assertCalled(0);

        map.messageArrived("", msg2);

        probe1.assertCalled(1);
        probe2.assertCalled(1);

        map.messageArrived("", msg1);

        probe1.assertCalled(2);
        probe2.assertCalled(1);
    }

    /**
     * Test action when connection failed
     *
     */
    @Test
    public void connectionFailedTest() {
        final MqttCallbackMap map = new MqttCallbackMap(str -> {});
        final ActionProbe failProbe = new ActionProbe();
        map.setConnectionFailedAction(failProbe);
        failProbe.assertCalled(0);
        map.connectionLost(new Throwable());
        failProbe.assertCalled(1);
    }


    private static class ActionProbe implements Runnable {

        private int nrofCalls = 0;

        @Override
        public void run() {
            nrofCalls++;
        }

        void assertCalled(int expected) {
            assertEquals("Incorrect number of calls", expected, nrofCalls);
        }
    }

}