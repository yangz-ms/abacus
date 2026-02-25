/**
 * Abacus — Internationalization data
 *
 * Calculator descriptions (calcN_desc) are provided by the backend registry
 * and merged at runtime via the /api/calculators endpoint.
 * This file only contains UI-chrome translations.
 *
 * To add a new language:
 *   1. Add an entry to the `languages` array (code, flag country-code, label)
 *   2. Add a matching translation object under `translations`
 *
 * Flag images are loaded from https://flagcdn.com/w40/{flag}.png
 */
window.ABACUS_I18N = {

  languages: [
    { code: "en", flag: "us", label: "English" },
    { code: "zh", flag: "cn", label: "\u4e2d\u6587" },
    { code: "hi", flag: "in", label: "\u0939\u093f\u0928\u094d\u0926\u0940" },
    { code: "es", flag: "es", label: "Espa\u00f1ol" },
    { code: "fr", flag: "fr", label: "Fran\u00e7ais" },
    { code: "ar", flag: "sa", label: "\u0627\u0644\u0639\u0631\u0628\u064a\u0629" },
    { code: "pt", flag: "br", label: "Portugu\u00eas" },
    { code: "ru", flag: "ru", label: "\u0420\u0443\u0441\u0441\u043a\u0438\u0439" },
    { code: "ja", flag: "jp", label: "\u65e5\u672c\u8a9e" },
    { code: "de", flag: "de", label: "Deutsch" }
  ],

  translations: {

    en: {
      title: "Abacus",
      tagline: "From arithmetic to algebra \u2014 beautifully rendered",
      calculate: "Calculate",
      input_label: "Input",
      result_label: "Result",
      error_label: "Error",
      placeholder: "Enter an expression\u2026",
      footer: "Powered by {0}",
      conn_error: "Connection error: ",
      group_expression: "Expression",
      group_solver: "Solver",
    },

    zh: {
      title: "Abacus",
      tagline: "\u4ece\u7b97\u672f\u5230\u4ee3\u6570 \u2014 \u7cbe\u7f8e\u5448\u73b0",
      calculate: "\u8ba1\u7b97",
      input_label: "\u8f93\u5165",
      result_label: "\u7ed3\u679c",
      error_label: "\u9519\u8bef",
      placeholder: "\u8f93\u5165\u8868\u8fbe\u5f0f\u2026",
      footer: "\u7531 {0} \u9a71\u52a8",
      conn_error: "\u8fde\u63a5\u9519\u8bef\uff1a",
      group_expression: "\u8868\u8fbe\u5f0f",
      group_solver: "\u6c42\u89e3\u5668",
    },

    hi: {
      title: "Abacus",
      tagline: "\u0905\u0902\u0915\u0917\u0923\u093f\u0924 \u0938\u0947 \u092c\u0940\u091c\u0917\u0923\u093f\u0924 \u0924\u0915 \u2014 \u0938\u0941\u0902\u0926\u0930 \u092a\u094d\u0930\u0938\u094d\u0924\u0941\u0924\u093f",
      calculate: "\u0917\u0923\u0928\u093e \u0915\u0930\u0947\u0902",
      input_label: "\u0907\u0928\u092a\u0941\u091f",
      result_label: "\u092a\u0930\u093f\u0923\u093e\u092e",
      error_label: "\u0924\u094d\u0930\u0941\u091f\u093f",
      placeholder: "\u090f\u0915 \u0935\u094d\u092f\u0902\u091c\u0915 \u0926\u0930\u094d\u091c \u0915\u0930\u0947\u0902\u2026",
      footer: "{0} \u0926\u094d\u0935\u093e\u0930\u093e \u0938\u0902\u091a\u093e\u0932\u093f\u0924",
      conn_error: "\u0915\u0928\u0947\u0915\u094d\u0936\u0928 \u0924\u094d\u0930\u0941\u091f\u093f: ",
      group_expression: "\u0905\u092d\u093f\u0935\u094d\u092f\u0915\u094d\u0924\u093f",
      group_solver: "\u0938\u0949\u0932\u094d\u0935\u0930",
    },

    es: {
      title: "Abacus",
      tagline: "De aritm\u00e9tica a \u00e1lgebra \u2014 bellamente renderizado",
      calculate: "Calcular",
      input_label: "Entrada",
      result_label: "Resultado",
      error_label: "Error",
      placeholder: "Ingresa una expresi\u00f3n\u2026",
      footer: "Impulsado por {0}",
      conn_error: "Error de conexi\u00f3n: ",
      group_expression: "Expresi\u00f3n",
      group_solver: "Solucionador",

    },

    fr: {
      title: "Abacus",
      tagline: "De l\u2019arithm\u00e9tique \u00e0 l\u2019alg\u00e8bre \u2014 magnifiquement rendu",
      calculate: "Calculer",
      input_label: "Entr\u00e9e",
      result_label: "R\u00e9sultat",
      error_label: "Erreur",
      placeholder: "Entrez une expression\u2026",
      footer: "Propuls\u00e9 par {0}",
      conn_error: "Erreur de connexion : ",
      group_expression: "Expression",
      group_solver: "Solveur",
    },

    ar: {
      title: "Abacus",
      tagline: "\u0645\u0646 \u0627\u0644\u062d\u0633\u0627\u0628 \u0625\u0644\u0649 \u0627\u0644\u062c\u0628\u0631 \u2014 \u0628\u0639\u0631\u0636 \u062c\u0645\u064a\u0644",
      calculate: "\u0627\u062d\u0633\u0628",
      input_label: "\u0627\u0644\u0645\u062f\u062e\u0644",
      result_label: "\u0627\u0644\u0646\u062a\u064a\u062c\u0629",
      error_label: "\u062e\u0637\u0623",
      placeholder: "\u0623\u062f\u062e\u0644 \u062a\u0639\u0628\u064a\u0631\u064b\u0627\u2026",
      footer: "\u0645\u062f\u0639\u0648\u0645 \u0628\u0640 {0}",
      conn_error: "\u062e\u0637\u0623 \u0641\u064a \u0627\u0644\u0627\u062a\u0635\u0627\u0644: ",
      group_expression: "\u062a\u0639\u0628\u064a\u0631",
      group_solver: "\u062d\u0644\u0651\u0627\u0644",

    },

    pt: {
      title: "Abacus",
      tagline: "Da aritm\u00e9tica \u00e0 \u00e1lgebra \u2014 lindamente renderizado",
      calculate: "Calcular",
      input_label: "Entrada",
      result_label: "Resultado",
      error_label: "Erro",
      placeholder: "Digite uma express\u00e3o\u2026",
      footer: "Desenvolvido com {0}",
      conn_error: "Erro de conex\u00e3o: ",
      group_expression: "Express\u00e3o",
      group_solver: "Solucionador",

    },

    ru: {
      title: "Abacus",
      tagline: "\u041e\u0442 \u0430\u0440\u0438\u0444\u043c\u0435\u0442\u0438\u043a\u0438 \u0434\u043e \u0430\u043b\u0433\u0435\u0431\u0440\u044b \u2014 \u043a\u0440\u0430\u0441\u0438\u0432\u043e \u043e\u0444\u043e\u0440\u043c\u043b\u0435\u043d\u043e",
      calculate: "\u0412\u044b\u0447\u0438\u0441\u043b\u0438\u0442\u044c",
      input_label: "\u0412\u0432\u043e\u0434",
      result_label: "\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
      error_label: "\u041e\u0448\u0438\u0431\u043a\u0430",
      placeholder: "\u0412\u0432\u0435\u0434\u0438\u0442\u0435 \u0432\u044b\u0440\u0430\u0436\u0435\u043d\u0438\u0435\u2026",
      footer: "\u0420\u0430\u0431\u043e\u0442\u0430\u0435\u0442 \u043d\u0430 {0}",
      conn_error: "\u041e\u0448\u0438\u0431\u043a\u0430 \u0441\u043e\u0435\u0434\u0438\u043d\u0435\u043d\u0438\u044f: ",
      group_expression: "\u0412\u044b\u0440\u0430\u0436\u0435\u043d\u0438\u0435",
      group_solver: "\u0420\u0435\u0448\u0430\u0442\u0435\u043b\u044c",

    },

    ja: {
      title: "Abacus",
      tagline: "\u7b97\u8853\u304b\u3089\u4ee3\u6570\u307e\u3067 \u2014 \u7f8e\u3057\u304f\u8868\u793a",
      calculate: "\u8a08\u7b97",
      input_label: "\u5165\u529b",
      result_label: "\u7d50\u679c",
      error_label: "\u30a8\u30e9\u30fc",
      placeholder: "\u5f0f\u3092\u5165\u529b\u2026",
      footer: "{0} \u3067\u52d5\u4f5c",
      conn_error: "\u63a5\u7d9a\u30a8\u30e9\u30fc\uff1a",
      group_expression: "\u5f0f",
      group_solver: "\u30bd\u30eb\u30d0\u30fc",

    },

    de: {
      title: "Abacus",
      tagline: "Von Arithmetik bis Algebra \u2014 wundersch\u00f6n dargestellt",
      calculate: "Berechnen",
      input_label: "Eingabe",
      result_label: "Ergebnis",
      error_label: "Fehler",
      placeholder: "Ausdruck eingeben\u2026",
      footer: "Betrieben mit {0}",
      conn_error: "Verbindungsfehler: ",
      group_expression: "Ausdruck",
      group_solver: "L\u00f6ser",

    }

  }
};
