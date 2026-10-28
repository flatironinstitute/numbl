import { useCallback, useEffect, useState } from "react";
import {
  checkRemoteServiceHealth,
  getRemoteServiceUrl,
  setRemoteServiceUrl as saveRemoteServiceUrl,
} from "../utils/remoteExecution";

function generatePasskey(): string {
  const key = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  sessionStorage.setItem("numbl_passkey", key);
  return key;
}

/**
 * Settings and health of the optional local execution server
 * (`numbl serve`), which runs IDE scripts instead of the browser worker.
 */
export function useRemoteExecution() {
  const [enabled, setEnabledState] = useState(false);
  const [serviceUrl, setServiceUrlState] = useState(getRemoteServiceUrl);
  const [passkey, setPasskey] = useState(
    () => sessionStorage.getItem("numbl_passkey") || generatePasskey()
  );

  const [nativeAddon, setNativeAddon] = useState<boolean | null>(null);
  const [serverStatus, setServerStatus] = useState<
    "unknown" | "connected" | "disconnected"
  >("unknown");

  const setEnabled = useCallback((on: boolean) => {
    setEnabledState(on);
    setNativeAddon(null);
    setServerStatus("unknown");
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    checkRemoteServiceHealth(serviceUrl, passkey).then(health => {
      if (cancelled) return;
      if (health) {
        setNativeAddon(health.nativeAddon === true);
        setServerStatus("connected");
      } else {
        setServerStatus("disconnected");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [enabled, serviceUrl, passkey]);

  /** Re-check the server at `url` (e.g. an unsaved draft URL). */
  const checkHealth = useCallback(
    (url: string) => {
      setServerStatus("unknown");
      checkRemoteServiceHealth(url, passkey).then(health => {
        if (health) {
          setNativeAddon(health.nativeAddon === true);
          setServerStatus("connected");
        } else {
          setServerStatus("disconnected");
        }
      });
    },
    [passkey]
  );

  const setServiceUrl = useCallback((url: string) => {
    setServiceUrlState(url);
    saveRemoteServiceUrl(url);
  }, []);

  const regeneratePasskey = useCallback(() => {
    setPasskey(generatePasskey());
    setServerStatus("disconnected");
  }, []);

  return {
    enabled,
    setEnabled,
    serviceUrl,
    setServiceUrl,
    passkey,
    nativeAddon,
    serverStatus,
    checkHealth,
    regeneratePasskey,
  };
}

export type RemoteExecution = ReturnType<typeof useRemoteExecution>;
