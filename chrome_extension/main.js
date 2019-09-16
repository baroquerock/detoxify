console.log('Detoxify is working');


const url = 'https://tatianag.pythonanywhere.com/predict'

const CATEGORIES = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 
                    'insult', 'threat', 'sexual_explicit'];

const CATEGORIES_HUMAN = {'severe_toxicity': 'severe toxicity', 'obscene': 'obscene language', 
                      'identity_attack': 'identity attack', 'insult' : 'insult',
                      'threat': 'threat', 'sexual_explicit': 'sexually explicit content'};


observers = {'detox': null}


const revealToxicity = selection => {

    document.querySelectorAll(selection).forEach(x => {

      let text = x.querySelector('#content-text');
      let container = x.querySelector('#content');
      let header = x.querySelector('#header');

      let div = document.createElement('div');
      let div_style = 'border: solid; min-height: 15px; border-width: thin; ' + 
                      'font-style: italic; font-size: 1.2em; text-align: center;'
      div.setAttribute('style', div_style); 
      div.setAttribute("id", "detox_container");
      
      let comment = {'comment': text.textContent};    

      $.post(url, comment, function(data) {

        if (data['success'] === 'true') {

          let tox = parseFloat(data['target']);

          let potential = [];

          for (let i = 1; i < CATEGORIES.length; i++) {
              data[CATEGORIES[i]] = parseFloat(data[CATEGORIES[i]]);
              if (data[CATEGORIES[i]] > 0.3) {
                let cat = CATEGORIES_HUMAN[CATEGORIES[i]]
                potential.push(cat)
              }
          };

          if (tox > 0.75) {
            let style = div.getAttribute('style')+'border-color: #D2222D;';
            div.setAttribute('style', style);
            container.setAttribute('style', 'background-color: #F2BDC0;'); 
          } 
          else if (tox > 0.3) {
            let style = div.getAttribute('style')+'border-color: #FFBF00;';
            div.setAttribute('style', style);
            container.setAttribute('style', 'background-color: #FFECB3;');
          } 
          else {
            let style = div.getAttribute('style')+'border-color: #238823;';
            div.setAttribute('style', style);
            container.setAttribute('style', 'background-color: #BDDBBD;'); 
          }


          let div_text = '';

          if (potential.length > 0) {
            div_text = div_text + ' - Potential ' + potential.join(", ");           
          }

          div_text = 'Toxicity index: ' + data['target'] + div_text;

          div.textContent = div_text;
          
          header.append(div);

        } else {

          let style = div.getAttribute('style')+'border-color: #E8E8E8;';
          div.setAttribute('style', style);
          container.setAttribute('style', 'background-color: #E8E8E8;');

          div.textContent = 'Our server has freewill, so he decided not to analyse this comment, but we are still negotiating!';
          header.append(div);

        }

      });

      x.classList.add("detoxified");
  });

};


const detoxity = commentsSection => {

  console.log('in detoxity')

  const selectors = ["#comment"];
  const withoutIndex = selectors.map(sel => `${sel}:not(.detoxified)`).join(", ");
  revealToxicity(withoutIndex);

  const mutationConfig = { attributes: false, childList: true, subtree: true };

  if (observers['detox'] !== null) {

    observers['detox'].observe(commentsSection, mutationConfig);

  } else {     

      const observer = new MutationObserver(() => {
        console.log('in MutationObserver');
        revealToxicity(withoutIndex);
      });

      observer.observe(commentsSection, mutationConfig);

      observers['detox'] = observer;
  };
  
};


const resetToxicity = () => {

  console.log('in resetToxicity')

  const selectors = ["#comment"];
  const selection = selectors.map(sel => `${sel}.detoxified`).join(", ");

  document.querySelectorAll(selection).forEach(x => {

      let container = x.querySelector('#content');
      let div = x.querySelector('#detox_container');

      if (container !== null) {
        container.setAttribute('style', 'background-color: #F9F9F9'); 
      } 

      if (div !== null) {
        div.remove();
        x.classList.remove("detoxified");
      } 
         
  });

};


const checkCommentsLoaded = () => {

  console.log('in checkCommentsLoaded')

  setTimeout(() => {
    // This selector is awful, but Youtube re-uses a lot of the DOM (the selector for the comments is re-used across a bunch of pages) so we need the exact path to the comments to match
    const commentsSection = document.querySelector(
      "html body ytd-app div#content.style-scope.ytd-app ytd-page-manager#page-manager.style-scope.ytd-app ytd-watch-flexy.style-scope.ytd-page-manager.hide-skeleton div#columns.style-scope.ytd-watch-flexy div#primary.style-scope.ytd-watch-flexy div#primary-inner.style-scope.ytd-watch-flexy ytd-comments#comments.style-scope.ytd-watch-flexy ytd-item-section-renderer#sections.style-scope.ytd-comments div#contents.style-scope.ytd-item-section-renderer"
    );

    if (commentsSection !== null) {

      chrome.runtime.onMessage.addListener(

        function(request, sender, sendResponse) {

          if (request.action == "reveal_toxicity") {
            console.log('revealing toxicity');
            detoxity(commentsSection);
            sendResponse();

          } else if (request.action == "reset") {
            console.log('resetting');
            if (observers['detox'] !== null) observers['detox'].disconnect()           
            resetToxicity();
            sendResponse();
          };
      });
    }
    else checkCommentsLoaded();
  }, 1000);
};




checkCommentsLoaded()

