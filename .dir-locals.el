((org-mode
  . ((org-hide-drawer-startup . t)
     (eval . (let* ((proj-root (locate-dominating-file default-directory ".dir-locals.el"))
                    (ob-fold (when proj-root
                               (expand-file-name "tools/ob-fold-hidden.el" proj-root))))
               (when (and ob-fold (file-readable-p ob-fold) (not (featurep 'ob-fold-hidden)))
                 (condition-case err
                     (load ob-fold nil t)
                   (error
                    (message "Failed to load ob-fold-hidden: %s" (error-message-string err)))))
               (when (fboundp #'el/org-setup-fold-hidden)
                 (add-hook 'org-mode-hook #'el/org-setup-fold-hidden nil t)
                 (el/org-setup-fold-hidden))))))

;; Identify problems with this code?
