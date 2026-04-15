/**
 * i18next initialization
 */
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from './en';
import es from './es';
import fr from './fr';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      es: { translation: es },
      fr: { translation: fr },
    },
    lng: 'en',
    fallbackLng: 'en',
    interpolation: { escapeValue: false },
  });

export default i18n;
